Published as a conference paper at ICLR 2026 

# ATTENTION, PLEASE! REVISITING ATTENTIVE PROBING THROUGH THE LENS OF EFFICIENCY 

**Bill Psomas**<sup>1</sup><sup>_∗_</sup> **Dionysis Christopoulos**<sup>2</sup><sup>_∗_</sup> **Eirini Baltzi**<sup>2</sup> **Ioannis Kakogeorgiou**<sup>7</sup> **Tilemachos Aravanis**<sup>1</sup> **Nikos Komodakis**<sup>3</sup><sup>_,_4</sup><sup>_,_5</sup> **Konstantinos Karantzalos**<sup>2</sup> 

**Yannis Avrithis**<sup>6</sup> **Giorgos Tolias**<sup>1</sup> 

1VRG, FEE, Czech Technical University in Prague 2National Technical University of Athens 3University of Crete 4Archimedes, Athena RC 5IACM-FORTH 6IARAI 

7IIT, NCSR “Demokritos” 

## ABSTRACT 

As fine-tuning becomes impractical at scale, probing is emerging as the preferred evaluation protocol. However, standard linear probing can understate the capability of models whose pre-training optimizes local representations rather than an explicit global representation. This motivates attentive probing, an alternative that uses attention to selectively aggregate patch-level features. Despite growing adoption, attentive probing is still underexplored: existing approaches are often overparameterized and computationally inefficient. In this work, we revisit attentive probing through the lens of the accuracy vs. parameter-efficiency trade-off. We present the first comprehensive study of existing methods, analyzing their design choices and benchmarking their performance. Building on these insights, we propose efficient probing (EP), a lightweight yet effective multi-query cross-attention mechanism that eliminates redundant projections and reduces the number of trainable parameters. Across multiple benchmarks and pre-training paradigms, EP consistently outperforms linear probing and previous attentive probing methods, and remains effective when combined with parameter-efficient fine-tuning. Beyond evaluation, our analysis uncovers emerging properties of EP, including complementary attention maps, which open new directions for leveraging probing beyond protocol design. Project page: https://vrg.fel.cvut.cz/ep/. 

## 1 INTRODUCTION 

The past few years have witnessed remarkable progress in _representation learning_ , with pre-training paradigms ranging from self-supervised learning (Chen et al., 2020; Caron et al., 2021; Kakogeorgiou et al., 2022), to vision–language models (Radford et al., 2021; Jia et al., 2021), and autoregressive architectures (El-Nouby et al., 2024; Fini et al., 2025). These diverse approaches share a _common goal_ : learning rich, transferable visual representations that minimize reliance on taskspecific labels and scale to large datasets. Evaluating the quality of such pre-trained representations is therefore _central_ to measuring progress. Conventional evaluation protocols include _k_ -NN classification, linear probing (LP), and full fine-tuning (FT). While _k_ -NN and LP assess the quality of the learned representations under a _frozen_ backbone, FT measures the utility of pre-training as _initialization_ for downstream tasks. Although FT achieves the highest performance, it is increasingly viewed as unsustainable and prohibitive at scale (Xin et al., 2024; Shuttleworth et al., 2024; Zou et al., 2023). As a result, probing is emerging as a practical evaluation protocol (Oquab et al., 2024; Bardes et al., 2024; Darcet et al., 2025). 

However, probing protocols remain _misaligned_ with many pre-training approaches. Standard LP typically relies on a single _global_ representation. e.g., the [CLS], which is well-suited for architectures trained with global objectives, but poorly reflects the potential of models such as masked image modeling (He et al., 2022), autoregressive (El-Nouby et al., 2024), or diffusion (Ma et al., 2024), where valuable information is distributed across _local_ representations. This gap motivates the rise of _attentive probing_ (Darcet et al., 2025; Chen et al., 2023; Bardes et al., 2024). Despite its promise, attentive probing remains underexplored. Existing methods vary significantly in design, 

> _∗_ Equal contribution 

1 

Published as a conference paper at ICLR 2026 

often suffering from _excessive parameterization_ and _inefficiency_ . Moreover, the connection between how attention aggregates features and why it improves predictive performance remains unclear. 

In this work, we address these limitations by conducting the first comprehensive study of attentive probing, revisiting its design through the lens of the _accuracy vs. parameter-efficiency trade-off_ . We introduce a unified framework that encompasses a wide range of attention-based aggregation methods—including those proposed for probing (El-Nouby et al., 2024; Bardes et al., 2024; Darcet et al., 2025) and others from unrelated tasks (Psomas et al., 2023; Noh et al., 2017; Rymarczyk et al., 2021). Through this framework, we derive _efficient probing_ (EP), a simple multi-query crossattention mechanism that eliminates redundant projections, reduces parameter count and computational cost, while matching or surpassing prior state-of-the-art performance. Moving beyond the parameter-efficient probing (PEP), we compare EP against parameter-efficient fine-tuning (PEFT) methods. We find that EP remains beneficial even when PEFT is allowed: combining EP with LoRA (Hu et al., 2022) yields configurations that dominate both pure probing and pure LoRA. 

Beyond efficiency and accuracy, a common ingredient of both existing methods and EP is the use of multiple independent _attention predictors_ (e.g., heads or queries). We show that a predictor’s contribution to classification accuracy correlates with its localization quality: predictors with sharper, foreground-focused attention drive larger accuracy gains. Rather than resorting to shortcut learning, e.g., leveraging background cues like water to classify a “fish”, EP’s predictors consistently attend to the object, improving interpretability, robustness, and performance. Notably, the attention maps of EP are more diverse and complementary than those of existing methods, with different predictors specializing in distinct object regions. Our contributions are threefold: 

1. We conduct the first systematic benchmark and analysis of attentive probing methods across diverse pre-training paradigms, comparing their accuracy, efficiency, and design choices. 

2. We introduce _efficient probing_ (EP), which achieves state-of-the-art accuracy, while bringing substantial gains in compute, memory, and parameter efficiency. 

3. We uncover a correlation between spatial localization and predictive performance, and show that EP produces diverse, complementary, and interpretable attention maps. 

## 2 RELATED WORK 

Self-supervised learning (SSL) has transformed visual representation learning, with evaluation typically performed via (i) _k-NN_ on frozen features, (ii) _linear probing_ (LP) using a shallow classifier on a frozen encoder, or (iii) _fine-tuning_ (FT) the entire model. Although FT achieves the highest accuracy, it is _computationally expensive_ , motivating the evaluation under frozen backbones. Two dominant SSL paradigms are joint embedding architectures (JEA) and masked image modeling (MIM). JEA methods (e.g., DINO (Caron et al., 2021)) contrast or cluster augmentations to learn global representations via a [CLS] token or pooled features. In contrast, MIM methods (e.g., MAE (He et al., 2022)) reconstruct masked regions, yielding localized, patch-distributed representations. This global vs. local distinction affects evaluation: LP is effective for JEA (Caron et al., 2021) but under-performs for MIM (He et al., 2022; Przewike´zlikowski et al., 2025), where discriminative information is not concentrated in a single token. 

Beyond SSL, _vision–language models_ (VLMs) pre-train on web-scale image–text corpora (e.g., LAION (Schuhmann et al., 2022)) and optimize cross-modal alignment (e.g., CLIP (Ilharco et al., 2021; Radford et al., 2021)). Although these models expose strong global descriptors, much of the signal remains distributed across patch tokens, making attentive aggregation appealing at probe time. Likewise, _auto-regressive_ (AR) families (e.g., AIM/AIMv2 (El-Nouby et al., 2024; Fini et al., 2025)) and _diffusion-based_ transformers (e.g., DiT (Peebles & Xie, 2023)) are primarily trained for generation (next-token prediction or denoising), not representation learning per se; nevertheless, their frozen features can be probed to assess representation quality. In all these cases, protocols that assume a single discriminative token may under-utilize the information spread across patches. 

To address this, recent work explores _attentive probing_ (El-Nouby et al., 2024; Chen et al., 2023; Bardes et al., 2024; Darcet et al., 2025), which learns attention to selectively aggregate patch tokens into informative descriptors for a linear classifier. While methods like AIM (El-Nouby et al., 2024), CAE (Chen et al., 2023), and V-JEPA (Bardes et al., 2024) adopt this idea, no unified evaluation exists. We fill this gap with a comprehensive benchmark and introduce a novel attention mechanism achieving a strong accuracy–efficiency trade-off. Additional related work appears in Appendix A. 

2 

Published as a conference paper at ICLR 2026 

## 3 METHOD 

### 3.1 ATTENTIVE POOLING 

**Preliminaries.** Let _X ∈_ R<sup>_Di×N_</sup> be the _feature matrix_ obtained from a _pre-trained_ and _frozen_ ViT backbone, where _Di_ is the number of feature channels and _N_ = _W × H_ the number of features, one per image patch across the spatial dimensions _W × H_ . Given the _input features X_ , the goal is to generate an _output image-level feature_ **_y_** _∈_ R<sup>_Do_</sup> by applying an _attentive pooling_ mechanism. The output feature is used to train a _C_ -way linear classifier with _Do_ ( _C_ + 1) parameters. 

We consider _M attention predictors_ , to be discussed in subsection 3.2. For each predictor _j ∈ {_ 1 _, . . . , M }_ , let **_a_** _j ∈_ R<sup>_N_</sup> be the _ℓ_ 1-normalized _attention vector_ it generates. Each vector, reshaped to _W × H_ , is an attention map indicating the locations on which the predictor focuses. Let _V ∈_ R<sup>_Do×N_</sup> be the _value features_ , commonly obtained by a linear transformation _V_ = _WV X_ , where _WV ∈_ R<sup>_Do×Di_</sup> is a learnable _projection matrix_ . 

Let the output feature **_y_** , value features _V_ and projection matrix _WV_ be partitioned into _M_ subvectors / submatrices according to 



with **_y_** _j ∈_ R<sup>_do_</sup> , _Vj ∈_ R<sup>_do×N_</sup> , _WVj ∈_ R<sup>_do×Di_</sup> and _do_ =<sup>_<u>D</u>_</sup> _M_<sup>_<u>o</u>_.</sup> 

The attentive pooling operation is then given by 



Each attention predictor is responsible for the weighted pooling of _N_ features into a _do_ -dimensional subspace of the final representation space. In the following, we explore existing and novel ways for designing these attention predictors. We focus on the number of additional parameters to be learnt on top of the frozen backbone and the computational complexity of the pooling operation. 

### 3.2 ATTENTION PREDICTORS 

**Multi-Head Cross-Attention (MHCA).** A vanilla approach is to perform multi-head crossattention between the input features and an _input vector_ **_u_** _∈_ R<sup>_Di_</sup> , where each head corresponds to a separate attention predictor. The _query feature_ **_q_** _∈_ R<sup>_Da_</sup> and _key features K ∈_ R<sup>_Da×N_</sup> are obtained by linear transformations **_q_** = _WQ_ **_u_** , _K_ = _WKX_ with projection matrices _WQ, WK ∈_ R<sup>_Da×Di_</sup> . 

Let the query feature **_q_** and projection matrix _WQ_ be partitioned into _M_ subvectors / submatrices according to 



with **_q_** _j_ = _WQj_ **_u_** _∈_ R<sup>_da_</sup> , _WQj ∈_ R<sup>_da×Di_</sup> and _da_ =<sup>_<u>D</u>_</sup> _M_<sup>_<u>a</u>_.Similarly,letthekeyfeatures</sup><sup>_K_and</sup> projection matrix _WK_ be partitioned according to 



with _Kj_ = _WKj X ∈_ R<sup>_da×N_</sup> and _WKj ∈_ R<sup>_da×Di_</sup> . 

The attention vector for head _j_ is then given by 



with 



3 

Published as a conference paper at ICLR 2026 



<!-- Start of picture text -->
softmax softmax<br>global  global<br>representation representation<br>patch tokens patch tokens<br><!-- End of picture text -->

Figure 1: _Comparison of multi-head cross-attention (MHCA, left) vs. our multi-query cross-attention (_ EP _, right)_ . MHCA uses an input vector **_u_** projected into query space and interacts with key features _K_ in (two) separate subspaces, each corresponding to an attention predictor. Attention predictor outputs **_a_** _j_ are used to aggregate value features _V_ into sub-vectors **_y_** _j_ , forming the final output **_y_** . In contrast, EP employs (two) _learnable queries_ **_q_** _j_ , one per attention predictor, to compute attention with input features directly in the full representation space. Attention predictor outputs **_a_** _j_ are used as in MHCA to perform the aggregation. 

That is, the input features _X_ and input vector **_u_** are projected to _da_ -dimensional subspaces where attention subvectors are computed via dot product followed by softmax normalization over patches. This attention predictor requires _Da_ (2 _Di_ + 1) parameters and has complexity _O_ ( _NDaDi_ ). As discussed in subsection 3.3, there are several existing methods that fit within this generic framework. 

**MHCA with a learnable query.** If we consider input vector **_u_** to be learnable, then there is no need for the projection matrix _WQ_ in (6). Instead, we can set the query feature **_q_** to be learnable, thus absorbing _WQ_ and **_u_** : 



where the query feature **_q_** _j ∈_ R<sup>_da_</sup> is learnable. 

We observe that _WK_<sup>_⊤_</sup> _j_<sup>maps</sup><sup>**_q_**</sup><sup>_j_to the</sup><sup>_Di_-dimensional space of input features to compute the atten-</sup> tion vector. Thus, standard MHCA ensures that each query subvector is interacting with the full representation space of the input features, despite being defined in a smaller dimensional space. Using a learnable query feature directly simplifies the architecture, reduces the amount of computations and the number of parameters to _Da_ ( _Di_ + 1). 

In order to explore ways for reducing the total number of parameters and to better understand the role of key transformation, we simplify the architecture by removing it. Letting _WK_ be fixed to the identity matrix, (7) becomes 



where the feature matrix _X_ is partitioned into _M_ submatrices according to 



with _Xj ∈_ R<sup>_di×N_</sup> and _di_ =<sup>_<u>D</u>_</sup> _M_<sup>_<u>i</u>_.Wethusobservethatthequeryfeatureonlyinteractswitha</sup><sup>_di_-</sup> dimensional subspace of the input features for _M >_ 1, which is a _limitation_ . We experimentally verify that setting _WK_ to identity matrix results in a noticeable performance drop. Based on this observation, in the following, we revisit (7) and design attention predictors that have less parameters and require less compute but do not experience a performance drop due to mathematical equivalence. 

**Parameter-efficient Multi-Query Cross-Attention (MQCA).** Instead of using the key submatrices _WKj_ to project the query subvectors **_q_** _j_ to the _Di_ -dimensional input feature space, we propose to learn _M_ effective query features **_u_** _j_ := _WK_<sup>_⊤_</sup> _j_<sup>**_q_**</sup><sup>_j∈_R</sup><sup>_Di_in that space directly (Figure 1).Thus,</sup><sup>**_u_**</sup><sup>_j_</sup> absorbs _WKj_ and **_q_** _j_ and attention prediction becomes 



for _j ∈{_ 1 _, . . . , M }_ . As a result, there are no projection matrices and there are no parameters other than the learnable query features **_u_** _j_ . 

4 

Published as a conference paper at ICLR 2026 

This choice reduces the number of additional parameters to be learned and saves from one more matrix-vector multiplication. In particular, it requires _DiM_ parameters for the attention compared to _Da_ ( _Di_ +1) for (6), while the number of operations drops to _NDiM_ compared to _NDa_ ( _Di_ +1). Typically, _M_ is one to two orders of magnitude smaller than _Di_ and _Da_ , which are commonly equal to each other, making the proposed approach more efficient in parameters and operations. 

There is a connection between EP and _slot attention_ (Locatello et al., 2020), where slots are also multiple vectors in the input feature space. To derive EP from slot attention, one needs to perform only a single iteration; remove LayerNorm, GRU and MLP; make slot vectors learnable rather than initialized at random; and concatenate the output features into a global representation of appropriate dimension. Thus, EP can be seen as a lightweight counterpart of slot attention, where the absence of interactions is compensated by the query features being learned. 

### 3.3 EXISTING VARIANTS 

We analyze existing methods as instances of the presented framework (Table 1), and examine common variants, considering their relationship to the framework despite slight deviations. Additional methods considered in experiments are presented in Appendix B. 

**AbMILP** (Rymarczyk et al., 2021) is the simplest variant. It fixes _WK_ and _WV_ to identity and is equivalent to MHCA with a learnable query feature framework in (7), with a single head ( _M_ = 1). It can also be seen as a special case of our proposed method in (10) with one learnable query feature, i.e. _M_ = 1. AbMILP requires only _Di_ parameters and computes attention with a single matrixvector multiplication, but its performance is limited by the single head/query. 

**AIM** (El-Nouby et al., 2024) is an instance of MHCA with a learnable query feature. It deviates from the generic framework by applying batch normalization on the input features. It does follow (2) and (7) with _M_ heads and _Da_ = _Di_ = _Do_ , but replaces _X_ by BN( _X_ ). Batch normalization introduces minor additional parameters and a slight computational overhead compared to the default variant of the framework. 

**DELF** (Noh et al., 2017) feeds each of the _N_ input features to a MLP whose output is a scalar attention value in [0 _,_ 1]. It can be viewed as an instance of the MHCA with a learnable query feature and _M_ = 1 with the following modifications. Key and value projection matrices share the same weights, i.e. _WK_ = _WV_ = _W_ , a non-linearity is introduced in equation (7) by **_a_** ˆ = ReLU( _WX_ )<sup>_⊤_</sup> **_q_** , where subscript _j_ is skipped due to _M_ = 1, and softmax in (5) is replaced by element-wise softplus, **_a_** = _σp_ (ˆ **_a_** ). In the context of DELF, the query feature **_q_** can be seen as the parameter of a 1 _×_ 1 convolutional layer. DOLG (Yang et al., 2021) adopts a similar design choice. 

**SimPool** (Psomas et al., 2023) can be seen as an instance of MHCA with a single head ( _M_ = 1) that uses a data-dependent input vector **_u_** , _WV_ fixed to identity, and layer normalization on the input features. Specifically, the query feature is obtained as **_q_** = _WQ_ **_u_** _,_ **_u_** = _N_<sup><u>1</u></sup><sup>_X⊤_</sup><sup>**1**(</sup><sup>_M_=1).</sup><sup>_X_</sup> is replaced by LN( _X_ ) for key and value transforms. Compared to MHCA with a learnable query, SimPool saves _Di_ parameters and has the same complexity. 

**V-JEPA.** The first part of V-JEPA (Bardes et al., 2024) is identical to the MHCA framework but applies layer normalization on the input features for key and value transforms, like SimPool. Its second part is an MLP with GeLU activation (Hendrycks & Gimpel, 2016) and residual connections, making the overall process equivalent to a transformer block. 

Table 1: _Attentive pooling variants as aligned algorithmic steps,_ fitting the framework of Sec. 3.1– 3.2. We list (i) how the query is formed, (ii) key/value transforms, (iii) attention predictor outputs, and (iv) pooling operation. _σm_ : softmax, _σp_ : softplus. _ϕ_ ( _x_ ) := _W_ 2 GeLU( _W_ 1 _x_ ). blue: learnable. 

|METHOD|QUERY SOURCE|KEY TRANSFORM|VALUE TRANSFORM|ATTENTION|POOLING|
|---|---|---|---|---|---|
|MHCA|**_q_**_j ∈_R<sup>_da_</sup>|_Kj_=_WKj X_|_Vj_=_WVj X_|**_a_**_j_=_σm_(_K_<sup>_⊤_</sup><br>_j_ <sup>**_q_**</sup><sup>_j_)</sup>|<sup>**_y_**</sup><sup>_j_=</sup><sup>_Vj_</sup><sup>**_a_**</sup><sup>_j_</sup>|
|AbMILP|**_q_** _∈_R<sup>_Di_</sup>|_K_=_X_|_V_=_X_|**_a_**=_σm_(_K_<sup>_⊤_</sup>**_q_**)|**_y_**=_V_**_a_**|
|AIM|**_q_**_j ∈_R<sup>_da_</sup>|_Kj_=_WKj_BN(_X_)|_Vj_=_WVj_BN(_X_)|**_a_**_j_=_σm_(_K_<sup>_⊤_</sup><br>_j_ <sup>**_q_**</sup><sup>_j_)</sup>|<sup>**_y_**</sup><sup>_j_=</sup><sup>_Vj_</sup><sup>**_a_**</sup><sup>_j_</sup>|
|DELF|**_q_** _∈_R<sup>_Di_</sup>|_K_=ReLU(_W X_)|_V_=_W X_|**_a_**=_σm_(_K_<sup>_⊤_</sup>**_q_**)|**_y_**=_V_**_a_**|
|SimPool|**_q_**=_WQ_**_u_** _∈_R<sup>_Di_</sup>_,_**_u_**= <sup>1</sup><br>_N _<sup>_X⊤_</sup><sup>**1**</sup>|<sup>_K_=</sup><sup>_WK_LN(</sup><sup>_X_)</sup>|_V_=LN(_X_)|**_a_**=_σm_(_K_<sup>_⊤_</sup>**_q_**)|**_y_**=_V_**_a_**|
|V-JEPA|**_q_**_j_=_WQj_**_u_** _∈_R<sup>_da_</sup>_,_**_u_** _∈_R<sup>_Di_</sup>|_Kj_=_WKj_LN(_X_)|_Vj_=_WVj_LN(_X_)|**_a_**_j_=_σm_(_K_<sup>_⊤_</sup><br>_j_ <sup>**_q_**</sup><sup>_j_)</sup>|<sup>**_y_**=</sup><sup>_ϕ_(</sup><sup>_WP Vj_</sup><sup>**_a_**</sup><sup>_j_)</sup>|
|EP (ours)|**_q_**_j ∈_R<sup>_Di_</sup>|_K_=_X_|_Vj_=_WVj X_|**_a_**_j_=_σm_(_K_<sup>_⊤_</sup>**_q_**_j_)|**_y_**_j_=_Vj_**_a_**_j_|



5 

Published as a conference paper at ICLR 2026 

## 4 EXPERIMENTS 

### 4.1 EXPERIMENTAL SETUP 

**Datasets.** We evaluate attentive probing across diverse image classification benchmarks, including ImageNet-1K (IN-1K) (Deng et al., 2009), CIFAR-100 (Krizhevsky et al., 2009), Places365 (Zhou et al., 2014), CUB-200 (Wah et al., 2011), FGVC Aircraft (Maji et al., 2013), Stanford Cars (Krause et al., 2013), and Food-101 (Bossard et al., 2014). More details in subsection C.1. 

**Pre-training methods.** We conduct attentive probing experiments with frozen models from diverse pre-training paradigms: MIM (e.g., MAE and CAPI), JEA (e.g., BYOL and DINO), hybrid (e.g., iBOT and DINOv2), VLMs (e.g., CLIP and SigLIP), and generative (e.g., DiT and AIMv2). Model sizes vary from small (e.g. ViT-S for MAE) to extra large (e.g. DiT-XL for DiT). 

**Pooling/probing methods.** We compare attentive probing against a diverse set of methods, covering different paradigms. First, we evaluate attentive poolings originally designed for probing, including AIM (El-Nouby et al., 2024), CAE (Chen et al., 2023), CAPI (Darcet et al., 2025), and V-JEPA (Bardes et al., 2024). Second, we include attentive poolings originally proposed in other contexts but applicable to probing, such as AbMILP (Rymarczyk et al., 2021), SimPool (Psomas et al., 2023), CLIP (Radford et al., 2021), SigLIP (Zhai et al., 2023; Wightman, 2019), CoCa (Yu et al., 2022), CaiT (Touvron et al., 2021), and DELF (Noh et al., 2017). Additionally, we include feature re-weighting methods like CBAM (Woo et al., 2018), applying global average pooling to obtain the global descriptor. As baselines, we use [CLS], which corresponds to standard linear probing when the backbone provides a classification token, and GAP, which serves as linear probing when no [CLS] token is available, but can also be interpreted as a baseline attentive probing method with uniform attention over patch tokens. To establish a reference, we also evaluate a ViT (Dosovitskiy et al., 2021) block, applying global average pooling to extract the global representation, and MHCA. All methods operate on the same input features—namely, the patch tokens extracted from the frozen backbone—ensuring a fair and consistent comparison. Unless otherwise stated, _Do_ = _Di_ = _Da_ . 

**Parameter-efficient fine-tuning (PEFT) methods.** We compare attentive probing against PEFT methods on ImageNet-1K: LoRA (Hu et al., 2022), BitFit (Ben Zaken et al., 2022), and LayerNorm tuning (Zhao et al.). 

**Evaluation protocols.** Attentive probing is performed for 90 epochs. We evaluate top-1 classification accuracy on the validation set of each dataset. Additionally, we compute the number of parameters and measure the FLOPs to assess computational efficiency and scalability. 

### 4.2 EXPERIMENTAL RESULTS 

**Accuracy vs. parameters.** In Figure 2, we compare efficient probing (EP) with baseline and competitor methods using MAE ViT-B, BEiTv2 ViT-B, and CAPI ViT-L on ImageNet-1K, and MAE ViT-B on Food-101. We plot top-1 accuracy against the number of trainable parameters, including both attentive pooling and classifier parameters, and overlay the Pareto frontier to highlight optimal trade-offs. The two primary baselines, [CLS] and GAP, are the most parameter-efficient, as they introduce no overhead beyond the classifier parameters, but yield noticeably lower accuracy. In contrast, methods like V-JEPA, CaiT, SigLIP, and the reference ViT block employ significantly more parameters, though within the attentive probing setting, their increased complexity provides mostly marginal accuracy improvements. Among the existing attentive probing or pooling methods, SimPool provides moderate accuracy but is not particularly parameter-efficient, while CAE and CLIP achieve stronger performance at the cost of higher parameter counts. AbMILP, DELF, AIM, and MHCA lie primarily on the Pareto frontier, striking the optimal balance. 

EP consistently achieves the best accuracy–parameter trade-off, positioning itself on the left or upper-left region of the Pareto frontier across pre-training methods and datasets. A key factor is its flexibility in controlling the number of queries _M_ and the output dimensionality _Do_ (because of _WV_ ), allowing adaptation to different parameter constraints. Notably, on ImageNet-1K with MAE ViT-B, EP64 (64 queries) achieves a state-of-the-art top-1 accuracy of 75.6% with less than 1.4M parameters. EP48 with 48 queries and _Do_ = _Di/_ 8 achieves 70.3% top-1 accuracy, while having a little more than 200k parameters, i.e. almost 4 _×_ less than linear probing ([CLS]). On ImageNet-1K with BEiTv2 ViT-B and with CAPI ViT-L, EP32 and EP64 achieve a state-of-the-art accuracy of 

6 

Published as a conference paper at ICLR 2026 



<!-- Start of picture text -->
(a) MAE ViT-B ImageNet-1K (b) MAE ViT-B Food-101<br>76 EPEP1696 EP EP EP16 64 4 AIM/MHCACAECLIP SigLIPV-JEPAViT 85 EP EP EP48 64 EPEP4 8 EP64AIM/MHCACoCa CAEV-JEPACaiTSigLIPViT<br>74 EP4 CaiT CLIP<br>EP64 EP2 DELF CoCa EP24 EP2 DELF<br>EP4 80 EP4<br>72 EP2 AbMILP SimPool EP2 AbMILP SimPool<br>EP48<br>70 EP8<br>EP2 Do = Di 75<br>Do = Di/ 2 CBAM Do = Di<br>68 [CLS] Do = Di/ 4 GAP Do = Di/ 4<br>GAP CBAM Do = Di/ 8 70 [CLS] D o = D i / 16<br>66<br>10 6 10 7 10 5 10 6 10 7<br>(c) BEiTv2 ViT-B ImageNet-1K (d) CAPI ViT-L ImageNet-1K<br>81 EP EP EP4 8 32 EPEPEP16 2 4 AbMILP DELFEP32AIM/MHCASimPoolCAECLIPV-JEPAViT 8482 EPEP648 EPEP864 AbMILP[CLS]EP EP 864 DELFEP EP 16 64 CoCaAIM/MHCACAECAPICLIPSigLIPCaiT ViTV-JEPA<br>EP2 CaiT<br>SigLIP<br>80<br>80 EP16<br>EP2<br>CoCa Do = Di Do = Di<br>78<br>79 [CLS] Do = Di/ 2 Do = Di/ 2<br>Do = Di/ 4 Do = Di/ 4<br>GAP GAP<br>Do = Di/ 8 Do = Di/ 8<br>CBAM 76<br>10 6 10 7 10 6 10 7<br>number of parameters number of parameters<br>top-1 accuracy (%)<br>top-1 accuracy (%)<br><!-- End of picture text -->

Figure 2: _Top-1 classification accuracy vs. number of parameters_ for various self-supervised pretraining methods across different datasets. We evaluate both dedicated probing mechanisms (e.g., V-JEPA) and repurposed attentive pooling methods (e.g., CLIP). EP variants are marked with different colors for different output dimensionalities _Do_ . EP _M_ : efficient probing with _M_ learnable queries. [CLS]: linear probing using the classification token; GAP: global average pooling over patch tokens; MHCA: multi-head cross-attention; ViT: default transformer block. 

81.7% and 83.7% respectively. On Food-101, a dataset smaller than ImageNet-1K, we observe that reducing _Do_ even to _Di/_ 16 does not significantly hurt performance. More results in subsection C.2. 

**Accuracy vs. computational cost.** In Figure 3, we compare different pooling/probing methods in terms of top-1 accuracy against computational cost, measured in GFLOPs. Baselines [CLS] and GAP are the most efficient but less accurate, while self-attention methods (e.g., ViT, CLIP) incur higher cost due to additional attention computations. EP achieves better accuracy than a ViT block with over 10 _×_ less compute, and consistently lies on the Pareto frontier, combining high accuracy with low cost. By adjusting the output dimensionality _Do_ , EP scales to different computational budgets without significant accuracy loss. Note that backbone FLOPs (e.g., 17.58 GFLOPs for MAE ViT-B) are not shown, since they are identical across all methods and thus omitted for fairness. Here we only compare the additional cost of the pooling/probing methods. 



<!-- Start of picture text -->
76 EP64 AIM/MHCA ViT<br>EP96<br>74 EP64 CoCa V-JEPACLIP<br>72 SimPool<br>AbMILP<br>70<br>Do = Di<br>68[CLS] Do = Di/ 2<br>GAP Do = Di/ 4<br>66<br>0 0 . 5 1 1 . 5<br>GFLOPs (beyond backbone)<br>top-1 accuracy (%)<br><!-- End of picture text -->

Figure 3: _Top-1 classification accuracy vs. GFLOPs_ for MAE ViT-B with different probings on ImageNet-1K. 

7 

Published as a conference paper at ICLR 2026 

Table 2: _Comparison of pre-training methods in terms of different evaluation protocols_ on ImageNet-1K. # PAR.: number of trainable parameters including classifier;<sup>_†_</sup> Results reported from the official paper under different than probing setup (e.g., augmentations);<sup>_‡_</sup> Evaluated using GAP. 

|METHOD<br>ARCH|PRE-TRAINING|k-NN|LP|# PAR.|EP|# PAR.|GAIN|FT<sup>_†_ </sup>|# PAR.|
|---|---|---|---|---|---|---|---|---|---|
|IM<br>MAE (He et al.,2022)<br>ViT-S/16<br>ViT-B/16<br>ViT-L/16|IN-1K|26.7<br>46.1<br>58.2|47.4<br>67.7<br>76.0|0.4M<br>0.8M<br>1.0M|64.6<br>75.6<br>79.3|0.5M<br>1.4M<br>2.1M|+17.2<br>+7.9<br>+3.3|80.6<br>83.6<br>85.9|22M<br>87M<br> 304M|
|M<br>BEiTv2 (Peng et al.,2022)<br>ViT-B/16<br>SimMIM (Xie et al.,2022)<br>ViT-B/16<br>CAPI (Darcet et al.,2025)<br>ViT-L/14|IN-1K<br>IN-1K<br>IN-1K|74.8<br>15.1<br>76.7|79.0<br>51.5<br>81.5|0.8M<br>0.8M<br>1.0M|81.7<br>65.1<br>83.6|1.4M<br>1.4M<br>2.1M|+2.7<br>+13.6<br>+2.1|85.0<br><br>83.8<br>_×_|87M<br>87M<br>304M|
|JEA<br>BYOL (Grill et al.,2020)<br>RN-50<br>DINO (Caron et al.,2021)<br>ViT-B/16|IN-1K<br>IN-1K|64.8<br>76.1|74.3<br>77.3|2.0M<br>0.8M|75.1<br>77.8|6.3M<br>1.4M|+0.8<br>+0.5|77.7<br>82.8|26M<br>87M|
|iBOT(Zhou et al.,2022)<br>ViT-B/16|IN-1K|77.0|78.7|0.8M|79.2|1.4M|+0.5|84.0|87M|
|BRID<br>DINOv2 (Oquab et al.,2024)<br>ViT-B/14<br>ViT-L/14|LVD-142M|81.8<br>83.5|83.2<br>85.2|0.8M<br>1.0M|84.0<br>85.6|1.4M<br>2.1M|+0.8<br>+0.4|_×_<br>_×_|87M<br>304M|
|Y<br>Franca(Venkataramanan et al.,2025) ViT-L/14|IN-21k|82.2|83.8|1.0M|84.3|2.1M|+0.5|_×_|304M|
|H<br>DINOv3 (Sim´eoni et al.,2025)<br>ViT-B/16<br>ViT-L/16|LVD-1689M|83.0<br>85.3|84.0<br>86.6|0.8M<br>1.0M|84.4<br>87.1|1.4M<br>2.1M|+0.4<br>+0.5|_×_<br>_×_|87M<br>304M|
|M<br>CLIP (Radford et al.,2021)<br>ViT-L/14|WIT|77.2|82.3<br>|0.8M|83.4|2.1M|+1.1|_×_|305M|
|VL<br>SigLIP (Zhai et al.,2023)<br>ViT-L/16<br>SigLIP2 (Tschannen et al.,2025)<br>ViT-L/16|WebLI<br>WebLI|83.7<br>84.4|84.1<sup>_‡_</sup><br>85.2<sup>_‡_</sup>|1.0M<br>1.0M|86.1<br>87.0|2.1M<br>2.1M|+2.0<br>+1.8|_×_<br>_×_|305M<br>305M|
|GEN<br>DiT (Peebles & Xie,2023)<br>DiT-XL/2<br>AIMv2 (Fini et al.,2025)<br>ViT-L/14|IN-1K<br>custom<sup>_∗_</sup>|8.3<br>80.8|32.7<sup>_‡_</sup><br>84.8<sup>_‡_</sup>|1.2M<br>1.0M|57.0<br>85.9|2.5M<br>2.1M|+24.3<br>+1.1|_×_<br>_×_|676M<br>304M|
|_Note._ custom<sup>_∗_</sup>: DFN-2B (Fang et al.,2024), COYO (<br>78|Byeon et al.,20|22), HQ|ITP (F<br>|ini et al.,|2025)|. Defau|lt EP=|EP32<br>Baseli<br>Baseli|.<br>ne probings<br>ne Pareto|
|68<br>70<br>72<br>74<br>76<br>EP<sup>_Di/_8</sup><br>~~EP~~<sup>_Di/_4</sup><br>top-1 accuracy (%)|AbMILP<br>EP<sup>_Di/_2</sup>||DELF<br>~~EP~~<sup>_D_</sup><sup>_~~i~~_</sup>|AI|M|||EP at v<br>EP Par<br>LoRA<br>LoRA<br>LoRA<br>LoRA<br>BitFit<br>LayerN<br>LoRA|arious_Do_<br>eto<br>at layer 4<br>at layer 8<br>at layer 12<br>on all layers<br>orm tuning<br>+EP|
||[CLS]|||||||LoRA|+EP Pareto|
|2_._5_·_10<sup>5</sup><br>5_·_10<sup>5</sup><br>number of p|10<br>arameters|<sup>6</sup>||2_·_|10<sup>6</sup>|||||



Figure 4: _Accuracy–parameter trade-off of probing and parameter-efficient fine-tuning methods_ on ImageNet-1K using a frozen MAE ViT-B/16 backbone. The _hybrid_ LoRA + EP configurations achieve a _strictly better trade-off_ than both pure EP and pure LoRA. 

**Comparison of pre-training methods.** Table 2 compares methods from diverse pre-training paradigms under multiple evaluation protocols on ImageNet-1K. Fine-tuning (FT) achieves the highest accuracy but is compute-intensive and unsustainable at scale, making it increasingly rare in recent evaluations ( _×_ ). _k_ -NN consistently underperforms, reflecting the weak separability of raw features. Linear probing (LP) offers a stronger baseline, yet EP provides consistent gains across all paradigms, often with only a small increase in parameters. Notably, EP changes the relative ranking of methods compared to LP and _k_ -NN: for example, MAE surpasses BYOL, and CAPI outperforms CLIP, challenging the view that MIM methods are weaker. In general, methods whose pre-training optimizes representations of patch tokens rather than an explicit global representation are benefit the most from attentive probing (e.g., SimMIM +13.6%, DiT +24.3%). We further analyze this trend in subsection C.3. Across the board, EP narrows the gap to full fine-tuning while remaining lightweight and scalable, demonstrating both its generality and effectiveness. 

**Comparison against PEFT methods.** In Figure 4 we compare the most efficient baseline probing methods ([CLS], AbMILP, DELF, AIM), EP at different output dimensionalities _Do_ , more than 40 LoRA variants, BitFit, and LayerNorm tuning, all on a frozen MAE ViT-B/16. We sweep LoRA across individual transformer layers (4, 8, 12) as well as all layers, and across multiple configurations (e.g. _WQ_ -only, _WK_ -only, _WV_ -only, _WQ_ + _WK_ , _WQ_ + _WK_ + _WV_ ) with ranks _ρ ∈{_ 8 _,_ 16 _,_ 32 _,_ 64 _}_ . 

Overall, LoRA applied to a small subset of layers (red, cyan, magenta crosses) lies roughly on or slightly above the baseline Pareto front, but is consistently dominated by EP in the accu- 

8 

Published as a conference paper at ICLR 2026 

racy–parameter plane. For instance, EP<sup>_Di/_2</sup> attains 74 _._ 9% top-1 accuracy with only 750K parameters, whereas the best single-layer LoRA configuration around that region requires around 1 _._ 2M parameters. LayerNorm tuning and BitFit (green and brown crosses) also outperform the pure probing baselines, yet EP remains strictly more efficient. All-layer LoRA configurations (orange crosses) move closer to the EP Pareto curve and some even surpass it in accuracy (up to 76 _._ 7%). In subsection C.2, we present an analysis with in- and out-of-domain _k_ -NN experiments on frozen features, showing that all-layer LoRA modifies the representation more strongly than EP, consistent with their roles as task-adaptive low-rank fine-tuning (LoRA) versus representation-preserving probing (EP). 

Motivated by these observations, we finally combine one of the most parameter-efficient LoRA variants (LoRA on all _WV_ matrices across layers) with EP for different output dimensionalities _Do_ (orange star–cross markers). The resulting LoRA+EP configurations form a new dominant region in the accuracy–parameter plane, _strictly improving over both pure EP and pure LoRA_ . For example, a hybrid setting with 850K parameters achieves 76 _._ 99% top-1 accuracy, improving over both the best pure EP configuration (75 _._ 58% with 1 _._ 38M parameters) and the best all-layer LoRA variant (76 _._ 72% with 1 _._ 95M parameters). At the low-parameter end, a LoRA+EP configuration with only 250K parameters already reaches 71 _._ 99% accuracy, i.e. about 4 _._ 3% above [CLS] linear probing (67 _._ 66%) while using over 3 _×_ fewer parameters. These results indicate that _EP captures information that LoRA alone does not_ , and vice versa: rather than being redundant with PEFT, EP provides a complementary, parameter-efficient probing (PEP) mechanism that remains beneficial. 

### 4.3 EXPERIMENTAL ANALYSIS 

**Impact of** _WK_ **and** _WV_ **.** Our analysis in subsection 3.2 posits that while a single learnable query **_q_** effectively absorbs the key transformation _WK_ in single-head attention, the same does not hold in multi-head. To empirically validate this, we probe MAE ViT-B pre-trained on ImageNet-1K with four variants: single-head ( _M_ =1) vs. multi-head ( _M>_ 1), each with (7) vs. without _WK_ (8). Specifically, we evaluate AbMILP and AIM for the single-head and multi-head (AIM12 with 12 heads) case, respectively. In single-head attention, removing _WK_ has minimal impact on performance (71.8% _→_ 71.7%), while in multi-head the drop is noticeable (75.1% _→_ 72.9%), since each query now interacts only with a subspace. Note that MHCA (7), that natively includes _WK_ , in AIM12 and our MQCA (10), that natively does not includes _WK_ , in EP12, configured with the same number of heads/queries ( _M_ =12), attain identical accuracy (75.1%), reflecting their mathematical equivalence. However, EP12 reaches this performance with far fewer parameters (1.36M for EP12 vs. 1.95M for AIM12), making its design more parameter-efficient. We also ablate the effect of the value transformation _WV_ , which operates on patch tokens (2), by adding or removing it across pooling methods. Introducing _WV_ to GAP results in a top-1 accuracy improvement from 66.7% _→_ 68.0%. Conversely, removing _WV_ from EP12 degrades performance from 75.1% _→_ 72.1%. A similar accuracy drop is observed for other methods, such as AIM (75.1% _→_ 72.0%) and CAE (74.9% _→_ 72.2%), confirming that _WV_ is a critical component. 



<!-- Start of picture text -->
2 . 5 2 . 5 3 3<br>2 2<br>1 . 5 1 . 5 2 2<br>1 AIM 1 AIM 1 Do = Di 1 Do = Di<br>0 . 5 EP 0 . 5 EP Do = Di/ 2 Do = Di/ 2<br>0 V-JEPA 0 V-JEPA 0 D o = D i / 4 0 D o = D i / 4<br>0 . 8 0 . 9 1 4 . 5 5 5 . 5 0 . 8 0 . 9 1 4 . 8 5 5 . 25 . 45 . 65 . 8 6<br>localization entropy localization entropy<br>∆accuracy<br><!-- End of picture text -->

Figure 5: _Classification accuracy vs. attention quality_ on ImageNet-1K. Each point corresponds to an attention predictor (head or query). ∆ accuracy measures the drop when replacing an attention predictor’s distribution with uniform. Plots show relations to localization quality (1st, 3rd) and entropy (2nd, 4th). Left: different attentive probing methods; Right: varying _Do_ for EP. 

**Classification vs. localization.** We investigate whether the quality of attention maps in terms of localization contributes positively to classification accuracy. Localization quality is measured by (i) the attention mass within the ground-truth bounding box (Deng et al., 2009) and (ii) the entropy of the attention distribution, averaged over the validation set. To estimate each predictor’s contribution 

9 

Published as a conference paper at ICLR 2026 

















Figure 6: _Attention maps of_ EP8. Each query is assigned a distinct color. Semantic correspondences emerge (e.g., tails, beaks, feet), with each query capturing complementary regions and enabling a structured decomposition of visual cues. MAE ViT-B pre-trained on IN-1K, probed with EP. 



<!-- Start of picture text -->
0 . 84<br>0 . 8 0 . 77 0 . 75 0 . 76 0 . 75 0 . 75<br>0 . 69 0 . 67 0 . 65<br>0 . 6 0 . 61 MHSA 0 . 55 0 . 59<br>0 . 44 0 . 47 0 . 46 V-JEPA<br>0 . 4 0 . 36 0 . 37 AIM<br>0 . 27 0 . 24 0 . 27 0 . 22 EP 0 . 24<br>0 . 2 0 . 18 0 . 16<br>0<br>MAE BEiTv2 SimMIM iBOT DINOv2 DINOv3 DINOv3 Franca CLIP SigLIP MAE<br>ViT-L ViT-B ViT-B ViT-B ViT-B ViT-B ViT-L ViT-L ViT-L ViT-L ViT-B<br>Complementarity<br><!-- End of picture text -->

Figure 7: _Complementarity scores of attention maps across different backbones and probings_ . We compare the diversity of the internal MHSA heads in the last block against the external EP queries (left), V-JEPA heads, and AIM heads (right). Number of predictors are matched for fairness. 

to classification, we measure the accuracy drop when replacing its attention with a uniform distribution. Figure 5 shows a clear correlation: predictors with better localization and lower entropy exert stronger influence on accuracy. This holds across different attentive methods as well as EP with reduced output dimensionality _Do_ . As also seen in Figure 2, lowering _Do_ degrades performance; the rightmost plots further suggest that this effect arises not only from reduced representational capacity but also from diminished attention quality, as reflected in higher entropy. 

**Attention complementarity.** In Figure 6, we jointly visualize the attention maps of EP8. An emerging property of EP is that its queries specialize in different object regions, yielding complementary and interpretable attention patterns. Queries consistently attend to distinct parts, producing semantic correspondences across images and a structured decomposition of visual cues. To quantify this diversity, we define a _complementarity_ metric measuring how differently attention predictors distribute mass over patch tokens. For standard backbones, we extract the [CLS] _→_ patch attention from the internal last-block MHSA heads, and for EP the learned queries, with matched counts for fairness. We _L_ 2-normalize each distribution, compute pairwise cosine similarities, and define complementarity as one minus the average off-diagonal similarity. Higher values indicate more diverse (less redundant) attention. As shown in Figure 7, EP achieves significantly higher complementarity than MHSA and outperforms other probing approaches. More experiments in subsection C.3. 

## 5 CONCLUSION 

We revisit evaluation protocols for pre-training methods and introduce efficient probing (EP), a scalable alternative to fine-tuning. Lightweight yet expressive, EP delivers interpretable attention, strong generalization across paradigms, and consistent gains over linear probing—up to +24.3% on ImageNet-1K. By comparing EP to standard parameter-efficient fine-tuning (PEFT) baselines, we show that EP is not only competitive on its own but also complementary: hybrid configurations achieve the best accuracy–parameter trade-offs, hinting at new research directions. 

**Acknowledgments.** This work was supported by the EU Horizon Europe programme MSCA PF RAVIOLI (No. 101205297) and the Junior Star GACR GM 21-28830M. We acknowledge VSB–Technical University of Ostrava, IT4Innovations National Supercomputing Center, Czech Republic, for awarding this project (OPEN-33-67) access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium, through the Ministry of Education, Youth and Sports of the Czech Republic via the e-INFRA CZ project (ID: 90254). AWS resources were provided by the National Infrastructures for Research and Technology GRNET and funded by the EU Recovery and Resiliency Facility. 

10 

Published as a conference paper at ICLR 2026 

## REFERENCES 

- Maximilian Augustin, Syed Shakib Sarwar, Mostafa Elhoushi, Yuecheng Li, Sai Qian Zhang, and Barbara De Salvo. Petah: Parameter efficient task adaptation for hybrid transformers. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , 2025. 17 

- Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mido Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. _Transactions on Machine Learning Research_ , 2024. ISSN 2835-8856. 1, 2, 5, 6, 17, 18 

- Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel. BitFit: Simple parameter-efficient finetuning for transformer-based masked language-models. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , 2022. 6, 17 

- Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 – mining discriminative components with random forests. In _European Conference on Computer Vision_ , 2014. 6, 19 

- Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset, 2022. 8 

- Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2021. 1, 2, 8, 19 

- Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, and Ping Luo. Adaptformer: Adapting vision transformers for scalable visual recognition. In _Advances in Neural Information Processing Systems_ , 2022. 17 

- Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In _International Conference on Machine Learning_ , 2020. 1 

- Xiaokang Chen, Mingyu Ding, Xiaodi Wang, Ying Xin, Shentong Mo, Yunhao Wang, Shumin Han, Ping Luo, Gang Zeng, and Jingdong Wang. Context autoencoder for self-supervised representation learning. _International Journal of Computer Vision_ , 132(1), 2023. 1, 2, 6, 18 

- Junsuk Choe, Seong Joon Oh, Seungho Lee, Sanghyuk Chun, Zeynep Akata, and Hyunjung Shim. Evaluating weakly supervised object localization methods right. In _Conference on Computer Vision and Pattern Recognition_ , 2020. 23 

- Junsuk Choe, Seong Joon Oh, Sanghyuk Chun, Seungho Lee, Zeynep Akata, and Hyunjung Shim. Evaluation for weakly supervised object localization: Protocol, metrics, and datasets. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2022. 23 

- Timoth´ee Darcet, Federico Baldassarre, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Cluster and predict latents patches for improved masked image modeling. _Transactions on Machine Learning Research_ , 2025. 1, 2, 6, 8, 19 

- Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _IEEE Conference on Computer Vision and Pattern Recognition_ , 2009. 6, 9, 19 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_ , 2021. 6, 17 

- Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel<sup>´</sup> Bautista, Vaishaal Shankar, Alexander T Toshev, Joshua M. Susskind, and Armand Joulin. Scalable pre-training of large autoregressive image models. In _Proceedings of the 41st International Conference on Machine Learning_ , 2024. 1, 2, 5, 6, 18 

11 

Published as a conference paper at ICLR 2026 

- Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander T Toshev, and Vaishaal Shankar. Data filtering networks. In _The Twelfth International Conference on Learning Representations_ , 2024. 8 

- Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor G Turrisi da Costa, Louis B´ethune, Zhe Gan, et al. Multimodal autoregressive pre-training of large vision encoders. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , 2025. 1, 2, 8, 20 

- Cheng Fu, Hanxian Huang, Xinyun Chen, Yuandong Tian, and Jishen Zhao. Learn-to-share: A hardware-friendly transfer learning framework exploiting computation and parameter sharing. In _Proceedings of the 38th International Conference on Machine Learning_ , 2021. 17 

- Jean-Bastien Grill, Florian Strub, Florent Altch´e, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. _Advances in Neural Information Processing Systems_ , 2020. 8, 19 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , 2016. 17 

- Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross B. Girshick. Masked autoencoders are scalable vision learners. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2022. 1, 2, 8, 19, 27 

- Xin He, Yushi Chen, Lingbo Huang, Danfeng Hong, and Qian Du. Foundation model-based multimodal remote sensing data classification. _IEEE Transactions on Geoscience and Remote Sensing_ , 62:1–17, 2024. 17 

- Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). _arXiv preprint arXiv:1606.08415_ , 2016. 5 

- Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In _International Conference on Learning Representations_ , 2022. 2, 6, 17 

- Leiyi Hu, Hongfeng Yu, Wanxuan Lu, Dongshuo Yin, Xian Sun, and Kun Fu. Airs: Adapter in remote sensing for parameter-efficient transfer learning. _IEEE Transactions on Geoscience and Remote Sensing_ , 62:1–18, 2024. 17 

- Chia-Chien Hung, Lukas Lange, and Jannik Str¨otgen. Tada: Efficient task-agnostic domain adaptation for transformers. In _The 61st Annual Meeting Of The Association For Computational Linguistics_ , 2023. 17 

- Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021. 2, 20 

- Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _International Conference on Machine Learning_ , 2021. 1 

- Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge J. Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In _European Conference on Computer Vision_ , 2022. 17 

- Ioannis Kakogeorgiou, Spyros Gidaris, Bill Psomas, Yannis Avrithis, Andrei Bursuc, Konstantinos Karantzalos, and Nikos Komodakis. What to hide from your students: Attention-guided masked image modeling. In _European Conference on Computer Vision_ , 2022. 1 

- Samar Khanna, Medhanie Irgau, David B. Lobell, and Stefano Ermon. Explora: Parameter-efficient extended pre-training to adapt vision transformers under domain shifts, 2025. 17 

12 

Published as a conference paper at ICLR 2026 

- Jonathan Krause, Michael Stark, Jia Deng, and Fei-Fei Li. 3d object representations for fine-grained categorization. _ICCVW_ , 2013. 6, 19 

- Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 6, 19 

- Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, et al. Matryoshka representation learning. _Advances in Neural Information Processing Systems_ , 2022. 27 

- Minglei Li, Peng Ye, Yongqi Huang, Lin Zhang, Tao Chen, Tong He, Jiayuan Fan, and Wanli Ouyang. Adapter-x: A novel general parameter-efficient fine-tuning framework for vision. _CoRR_ , abs/2406.03051, 2024. 17 

- Tao Li, Zhengbao He, Yujun Li, Yasheng Wang, Lifeng Shang, and Xiaolin Huang. Flat-lora: Lowrank adaptation over a flat loss landscape, 2025. 17 

- Min Lin, Qiang Chen, and Shuicheng Yan. Network in network. _arXiv preprint arXiv:1312.4400_ , 2013. 17 

- Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. _Advances in Neural Information Processing Systems_ , 2020. 5 

- Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In _European Conference on Computer Vision_ , 2024. 1 

- S. Maji, J. Kannala, E. Rahtu, M. Blaschko, and A. Vedaldi. Fine-grained visual classification of aircraft. Technical report, 2013. 6, 19 

- Hyeonwoo Noh, Andre Araujo, Jack Sim, Tobias Weyand, and Bohyung Han. Large-scale image retrieval with attentive deep local features. In _Proceedings of the IEEE International Conference on Computer Vision_ , 2017. 2, 5, 6, 17 

- Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. _Transactions on Machine Learning Research_ , 2024. 1, 8, 19 

- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. _Advances in Neural Information Processing Systems_ , 2019. 20 

- William Peebles and Saining Xie. Scalable diffusion models with transformers. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2023. 2, 8, 20 

- Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, and Furu Wei. Beit v2: Masked image modeling with vector-quantized visual tokenizers. _CoRR_ , abs/2208.06366, 2022. 8, 19 

- Jonas Pfeiffer, Aishwarya Kamath, Andreas R¨uckl´e, Kyunghyun Cho, and Iryna Gurevych. Adapterfusion: Non-destructive task composition for transfer learning. In _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics_ , 2021. 17 

- Marcin Przewike´zlikowski, Randall Balestriero, Wojciech Jasi´nski, Marek Smieja,<sup>´</sup> and Bartosz Zieli´nski. Beyond [cls]: Exploring the true potential of masked image modeling representations. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2025. 2, 27 

- Bill Psomas, Ioannis Kakogeorgiou, Konstantinos Karantzalos, and Yannis Avrithis. Keep it simpool: Who said supervised transformers suffer from attention deficit? In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2023. 2, 5, 6, 17 

13 

Published as a conference paper at ICLR 2026 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _ICML_ , Proceedings of Machine Learning Research, 2021. 1, 2, 6, 8, 17, 18, 19 

- Dawid Rymarczyk, Adriana Borowa, Jacek Tabor, and Bartosz Zieli´nski. Kernel self-attention in deep multiple instance learning, 2021. 2, 5, 6, 17, 18 

- Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. _Advances in Neural Information Processing Systems_ , 2022. 2 

- Reece Shuttleworth, Jacob Andreas, Antonio Torralba, and Pratyusha Sharma. Lora vs full finetuning: An illusion of equivalence. _arXiv preprint arXiv:2410.21228_ , 2024. 1 

- Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, et al. Dinov3. _arXiv preprint arXiv:2508.10104_ , 2025. 8 

- Romain Thoreau, Valerio Marsocci, and Dawa Derksen. Parameter-efficient adaptation of geospatial foundation models through embedding deflection. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2025. 17 

- Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Herv´e J´egou. Going deeper with image transformers. In _IEEE/CVF International Conference on Computer Vision_ , 2021. 6, 17, 18 

- Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. _arXiv preprint arXiv:2502.14786_ , 2025. 8, 19 

- Shashanka Venkataramanan, Valentinos Pariza, Mohammadreza Salehi, Lukas Knobel, Spyros Gidaris, Elias Ramzi, Andrei Bursuc, and Yuki M Asano. Franca: Nested matryoshka clustering for scalable visual representation learning. _arXiv preprint arXiv:2507.14137_ , 2025. 8 

- Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The CaltechUCSD Birds-200-2011 Dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011. 6, 19 

- Ross Wightman. Pytorch image models, 2019. 6, 17, 18 

- Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon. CBAM: convolutional block attention module. In _European Conference on Computer Vision_ , 2018. 6, 17 

- Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: a simple framework for masked image modeling. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2022. 8, 19 

- Yi Xin, Siqi Luo, Haodi Zhou, Junlong Du, Xiaohong Liu, Yue Fan, Qing Li, and Yuntao Du. Parameter-efficient fine-tuning for pre-trained vision models: A survey. _arXiv e-prints_ , pp. arXiv– 2402, 2024. 1 

- Min Yang, Dongliang He, Miao Fan, Baorong Shi, Xuetong Xue, Fu Li, Errui Ding, and Jizhou Huang. Dolg: Single-stage image retrieval with deep orthogonal fusion of local and global features. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2021. 5, 17 

- Yang You, Igor Gitman, and Boris Ginsburg. Large batch training of convolutional networks. _arXiv preprint arXiv:1708.03888_ , 2017. 20 

14 

Published as a conference paper at ICLR 2026 

- Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. _Transactions on Machine Learning Research_ , 2022. 6, 17, 18 

- Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In _IEEE/CVF International Conference on Computer Vision_ , 2023. 6, 8, 18, 19 

- Bingchen Zhao, Haoqin Tu, Chen Wei, Jieru Mei, and Cihang Xie. Tuning layernorm in attention: Towards efficient multi-modal llm finetuning. In _The Twelfth International Conference on Learning Representations_ . 6, 17 

- Houqiang Zhong, Shaocheng Shen, Ke Cai, Zhenlong Wu, Jiangchao Yao, Yuan Cheng, Xuefei Li, Xiaoyun Zhang, Li Song, and Qiang Hu. Serial low-rank adaptation of vision transformer. In _IEEE International Conference on Multimedia and Expo_ , 2025. 17 

- Bolei Zhou, Agata Lapedriza, Jianxiong Xiao, Antonio Torralba, and Aude Oliva. Learning deep features for scene recognition using places database. In _Advances in Neural Information Processing Systems_ , 2014. 6, 19 

- Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. _International Conference on Learning Representations_ , 2022. 8, 19 

- Yitao Zhu, Zhenrong Shen, Zihao Zhao, Sheng Wang, Xin Wang, Xiangyu Zhao, Dinggang Shen, and Qian Wang. Melo: Low-rank adaptation is better than fine-tuning for medical image diagnosis. In _IEEE International Symposium on Biomedical Imaging_ , 2024. 17 

- Wentao Zou, Qi Li, Jidong Ge, Chuanyi Li, Xiaoyu Shen, Liguo Huang, and Bin Luo. A comprehensive evaluation of parameter-efficient fine-tuning on software engineering tasks. _arXiv preprint arXiv:2312.15614_ , 2023. 1 

15 

Published as a conference paper at ICLR 2026 

|AP|PENDIX||
|---|---|---|
|**A **|**Additional Related Works**|**17**|
||A.1<br>Pooling<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>17|
||A.2<br>Parameter-efficient Fine-tuning . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>17|
||Additive. . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>17|
||Selective. . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>17|
||Low-rank adaptation. . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>17|
|**B**|**Additional Methods**|**18**|
||CLIP. . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>18|
||CAiT. . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>18|
||SigLIP. . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>18|
||CAE. . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>18|
||CoCa . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>18|
|**C **|**Additional Experiments**|**19**|
||C.1<br>Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>19|
||Datasets.<br>. . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>19|
||Pooling/probing methods.<br>. . . . . . . . . . . . . . . . .|. . . . . . . . .<br>19|
||Implementation details. . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>19|
||C.2<br>Experimental Results . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>20|
||Accuracy vs. parameters. . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>20|
||Layer-wise probing.<br>. . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>21|
||Low-shot probing.<br>. . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>22|
||In- and out-of-domain k-NN evaluation. . . . . . . . . . .|. . . . . . . . .<br>22|
||Object localization. . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>23|
||Image retrieval. . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>23|
||C.3<br>Experimental Analysis<br>. . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>23|
||Attention complementarity.. . . . . . . . . . . . . . . . .|. . . . . . . . .<br>23|
||Attention complementarity as a loss. . . . . . . . . . . . .|. . . . . . . . .<br>24|
||Entropy. . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>26|
||Impact of attention predictors,_Da_, and_Do_. . . . . . . . .|. . . . . . . . .<br>26|
||EP convergence.<br>. . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>27|
||Attention mixing. . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>27|
||Matryoshka representation learning. . . . . . . . . . . . .|. . . . . . . . .<br>27|
||Visualizations.<br>. . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>28|
|**D **|**Limitations and Future Work**|**32**|
|**E**|**Use of Large Language Models**|**32**|



16 

Published as a conference paper at ICLR 2026 

## A ADDITIONAL RELATED WORKS 

### A.1 POOLING 

Pooling reduces spatial resolution while retaining semantic information. In CNNs, fixed pooling (e.g., global average pooling (Lin et al., 2013; He et al., 2016)) is standard; in vision transformers (ViTs) (Dosovitskiy et al., 2021), the [CLS] token aggregates features via self-attention. 

Recent work proposes attention-based pooling to enhance representation quality. SimPool (Psomas et al., 2023) replaces global average pooling using trainable attention in both CNNs and ViTs. Vision-language models such as CLIP (Radford et al., 2021), SigLIP (Wightman, 2019), and CoCa (Yu et al., 2022) use attentive pooling or cross-attention to fuse modalities. V-JEPA (Bardes et al., 2024) applies cross-attention pooling for probing pretrained representations. In image retrieval, DELF (Noh et al., 2017) and DOLG (Yang et al., 2021) use spatial attention to focus on salient regions. CaiT (Touvron et al., 2021) improves class-token attention, AbMILP (Rymarczyk et al., 2021) uses single-query pooling for multiple-instance learning, and CBAM (Woo et al., 2018) combines channel and spatial attention to recalibrate features. Although these poolings are originally introduced in diverse contexts, we repurpose them for probing of frozen pre-trained models, enabling a fair and comprehensive benchmark. 

### A.2 PARAMETER-EFFICIENT FINE-TUNING 

Parameter-efficient fine-tuning (PEFT) adapts large pre-trained models to downstream tasks without updating all model parameters. PEFT techniques broadly include _additive_ , _selective_ , and _low-rank adaptation_ methods. 

**Additive.** Additive methods introduce small, task-specific modules into the frozen backbone, leaving the pre-trained weights untouched. These modules often reside within the transformer blocks and are trained to specialize the model for a new domain. Notable examples include AdapterFusion (Pfeiffer et al., 2021), LeTS (Fu et al., 2021), and TADA (Hung et al., 2023) in natural language processing (NLP), VPT (Jia et al., 2022), AdaptFormer (Chen et al., 2022), and AdapterX (Li et al., 2024) in computer vision (CV), and FMA (He et al., 2024), AiRs (Hu et al., 2024), and DEFLECT (Thoreau et al., 2025) in remote sensing (RS). 

**Selective.** Selective methods fine-tune only specific subsets of parameters, typically chosen based on their functional role or estimated importance. Examples include BitFit (Ben Zaken et al., 2022), which adjusts only the bias terms, and norm-tuning approaches (Zhao et al.) which update only the normalization layers. These techniques avoid introducing new components, making them lightweight, though sometimes at the expense of performance. 

**Low-rank adaptation.** Low-rank adaptation methods like LoRA (Hu et al., 2022) in NLP assume that parameter updates lie in a low-dimensional subspace. They inject trainable low-rank matrices into existing layers, yielding strong performance with minimal parameter growth. In the vision domain, LoRa and its variants have been effectively adapted to vision transformers (ViTs), often rethinking where and how low-rank modules are inserted to align with the spatial and hierarchical nature of visual representations. Notable examples include structure-aware methods like Serial LoRA (Zhong et al., 2025) and Flat-LoRA (Li et al., 2025), layer-wise extensions such as AdaptFormer (Chen et al., 2022), and task specific designs like PETAH (Augustin et al., 2025) and MeLo (Zhu et al., 2024), which adapt LoRa to mobile inference and medical imaging, respectively. Continued pretraining approaches such as ExPLoRA (Khanna et al., 2025) further extend low-rank adaptation to domain-shifted self-supervised settings. 

EP naturally fits the additive PEFT family. It introduces a compact learnable query set interacting with frozen tokens via multi-head cross-attention. Unlike typical prompt-based methods, it avoids backbone modifications and focuses training on minimal parameters. Thus, EP efficiently combines additive PEFT simplicity with task-specific attentive pooling. 

17 

Published as a conference paper at ICLR 2026 

## B ADDITIONAL METHODS 

We provide here additional attentive pooling/probing methods evaluated in subsection 4.2 but not described in detail in subsection 3.3. These approaches represent variations of the MHCA framework, highlighting only their key deviations from the default design. 

**CLIP.** CLIP (Radford et al., 2021) differs from MHCA by employing self-attention rather than cross-attention. Specifically, CLIP prepends a global average pooled (GAP) token to the layernormalized input features, treating this token as a global representation. All tokens, including the GAP token, are augmented with learnable positional encodings and processed through a single self-attention block (which includes a query projection matrix _WQ_ ). The global representation is extracted from the output corresponding to the GAP token. Additionally, CLIP includes a linear projection matrix _WP_ after attention aggregation. These modifications enable interactions across all tokens but increase parameter count and computational complexity. 

**CAiT.** CAiT (Touvron et al., 2021) adapts the MHCA-with-learnable-query formulation by concatenating the learnable query token with the input features and applying self-attention rather than cross-attention. It retains the query projection matrix _WQ_ and includes a linear projection matrix _WP_ after attention aggregation, followed by an MLP block with GELU activations, residual connections, and LayerScale parameters, similar to V-JEPA. The global representation is obtained from the updated query token after these operations, thus increasing complexity and parameter count relative to the default MHCA variant. 

**SigLIP.** SigLIP (Zhai et al., 2023; Wightman, 2019) remains close to the MHCA-with-learnablequery formulation but retains the query projection matrix _WQ_ . After the attention aggregation, SigLIP incorporates an output projection _WP_ , followed by a transformer-style MLP block with GELU activation and residual connections, similar to V-JEPA and CAiT. Optional layer normalization can also be applied before the MLP. These changes add further parameters and computational overhead compared to the baseline MHCA design. 

**CAE.** CAE (Chen et al., 2023) follows the MHCA-with-learnable-query template closely but retains the query projection matrix _WQ_ and applies separate layer normalization to both input features and the query token prior to attention. After attention aggregation, it employs an additional output projection matrix _WP_ . These modifications introduce additional parameters and computational complexity. 

**CoCa** CoCa (Yu et al., 2022) is aligned with the MHCA-with-learnable-query framework but retains the query projection matrix _WQ_ and layer-normalizes the query token before computing attention. Attention and value aggregation both occur in a reduced-dimensional space, with dimension _Da_ = _Do < Di_ . A final linear projection matrix _W_ proj is then applied to restore the feature dimension to the original backbone dimension _Di_ . These choices introduce a controlled amount of additional complexity and parameters. 

In Figure 8 we present a visual comparison of three selected attentive pooling/probing techniques: AbMILP (Rymarczyk et al., 2021), AIM (El-Nouby et al., 2024), and V-JEPA (Bardes et al., 2024). AbMILP (top-left) serves as a lightweight method, employing a single-head learnable query without additional linear projection matrices, thus requiring only _Di_ parameters. AIM (top-right) extends this by adopting multi-head cross-attention, operating within multiple subspaces. This approach introduces linear projection matrices for keys and values, increasing the number of parameters, yet allowing more expressive query-key interactions. V-JEPA (bottom) represents a significantly more complex and computationally intensive architecture. Beyond multi-head attention and multiple linear projections for queries, keys, and values, it integrates an additional projection step, followed by a multi-layer perceptron (MLP) featuring GeLU activation and residual connections. 

18 

Published as a conference paper at ICLR 2026 



Figure 8: _Visual comparison of three attentive pooling/probing methods_ . AbMILP (top-left) employs a single-head, learnable query without linear projections, minimizing complexity. AIM (topright) extends the approach by introducing multi-head attention, operating in multiple subspaces, and applies linear projections to keys and values. V-JEPA (bottom) offers a more comprehensive architecture by integrating multi-head attention with extensive linear projections and an additional MLP block with a residual connection, increasing representational capacity. 

## C ADDITIONAL EXPERIMENTS 

### C.1 EXPERIMENTAL SETUP 

**Datasets.** We evaluate attentive probing across diverse image classification benchmarks. As a large-scale dataset, ImageNet-1K (Deng et al., 2009) serves as the primary benchmark, containing 1.28M images across 1,000 categories. CIFAR-100 (Krizhevsky et al., 2009) provides a smaller yet challenging 100-class task with 60K images. To assess scene understanding, we use Places365 (Zhou et al., 2014), comprising 1.8M images spanning 365 scene types. For fine-grained classification, we evaluate on CUB-200 (Wah et al., 2011) (11,788 images, 200 bird species), FGVC Aircraft (Maji et al., 2013) (10K images, 100 aircraft models), Stanford Cars (Krause et al., 2013) (16K images, 196 car types), and Food-101 (Bossard et al., 2014) (101K images, 101 food categories). Together, these datasets span a wide spectrum of scales and challenges—large-scale vs. small-scale, generic vs. fine-grained, and object- vs. scene-centric—providing a comprehensive testbed for probing methods. 

**Pooling/probing methods.** We adopt AbMILP Table 3: _AbMILP architecture ablation_ increaswith depth=1, which reduces AbMILP to the ing MLP depth. # PAR.: number of parameters. formulation described in Table 1 and subsection 3.3. In Table 3, we evaluate AbMILP on METHOD # PAR. ACCURACY ImageNet-1K with larger depths (depth=2,3), AbMILP (depth 1) 769,769 71.74 which correspond to the MLP-based variants AbMILP (depth 2) 1,360,361 72.25 and scale as _O_ ( _Di_<sup>2),exploringtheirparame-</sup> AbMILP (depth 3) 1,950,953 72.84 ter–accuracy tradeoffs. Although deeper MLPs introduce substantially more parameters (+ _O_ ( _Di_<sup>2)), the gains in accuracy are marginal.To ensure a</sup> fair comparison in the _accuracy-vs.-parameter-efficiency_ setting, we consider the most competitive AbMILP variant (depth=1) as default, which successfully lies on the Pareto front. 

**Implementation details.** We evaluate 15 models spanning five pre-training paradigms: four masked image modeling (MAE (He et al., 2022), BEiTv2 (Peng et al., 2022), SimMIM (Xie et al., 2022), CAPI (Darcet et al., 2025)), two joint-embedding (BYOL (Grill et al., 2020), DINO (Caron et al., 2021)), two hybrid (iBOT (Zhou et al., 2022), DINOv2 (Oquab et al., 2024)), three visionlanguage (CLIP (Radford et al., 2021), SigLIP (Zhai et al., 2023), SigLIP2 (Tschannen et al., 2025)), 

19 

Published as a conference paper at ICLR 2026 

and two generative—DiT (diffusion) (Peebles & Xie, 2023) and AIMv2 (autoregressive) (Fini et al., 2025). Architectures range from small (e.g., ViT-S for MAE) to extra-large (e.g., DiT-XL for DiT). 

To ensure a fair comparison of pooling/probing methods, we mostly adopt the LARS optimizer You et al. (2017) and conduct a learning rate search in the range [0.1, 5.0] with a step size of 0.1 for each model. For large-scale datasets such as ImageNet-1K and Places365, we fix the learning rate to 0.1 due to the computational cost of an exhaustive search. All models are trained for 90 epochs with 10 warm-up epochs, ensuring consistent training schedules even though many models converge much earlier (e.g., SigLIP within 15 epochs). The effective batch size is set to 4096 for all datasets except FGVC Aircraft, where it is reduced to 512 due to the smaller dataset size. Data augmentation follows standard PyTorch (Paszke et al., 2019) image pre-processing: RandomResizedCrop, horizontal flipping, and normalization. For vision-language models, we adopt their official preprocessing pipelines (e.g., OpenCLIP (Ilharco et al., 2021) transforms for CLIP, SigLIP, and SigLIP2) to ensure alignment with pre-training distributions. Experiments are conducted on a cluster of 8 NVIDIA A100 GPUs (40 GB VRAM each) and on the LUMI supercomputer, on clusters of 8 AMD Instinct MI250X GPUs (128 GB HBM2e memory each). 

### C.2 EXPERIMENTAL RESULTS 

**Accuracy vs. parameters.** We extend the benchmark presented on subsection 4.2 to include additional pre-training methods and datasets. Figure 9 and Figure 10 present the trade-off between top-1 accuracy and the number of trainable parameters (including the classifier) for various pooling/probing methods integrated into MAE and SimMIM with different backbone sizes (ViT-S, ViT-B and ViT-L). The evaluation spans multiple datasets, including FGVC-Aircraft, CUB200, Places365, CIFAR-100 and Cars196. 

As shown in Figure 9a and Figure 9b, EP consistently outperforms standard linear probing across MAE ViT-S and ViT-L. Notably, on MAE ViT-L, EP16 with _Do_ = _Do/_ 2 achieves an accuracy boost of 79.1% surpassing linear probing by 3.1% while maintaining the same number of trainable parameters. Furthermore, EP128 reaches 79.4%, outperforming SigLIP, while reducing the number of trainable parameters by over 11M. 

In Figure 9c, we benchmark attentive probing on SimMIM ViT-B pre-trained on ImageNet-1K. Baselines such as [CLS] and GAP remain parameter-efficient but yield relatively low accuracy. Classical attention modules like CBAM and AbMILP do not improve this trade-off, while methods such as ViT, V-JEPA, and SigLIP achieve higher accuracy but at the cost of orders-of-magnitude more parameters. EP strikes a favorable balance: scaling the number of queries (EP2–EP64) consistently increases accuracy, while reducing _Do_ effectively lowers parameter count with only moderate drops in performance. Notably, EP2 with _Do_ = _Di/_ 2 achieves 60.6% top-1 accuracy using fewer than 0.7M parameters, outperforming GAP and DELF under similar budgets. At the high end, EP64 reaches 65.1% accuracy, closing the gap to heavy-weight probing methods while remaining lighter. 

In Figure 9d, our EP24 variant, for the FGVC-Aircraft dataset, achieves a remarkable accuracy boost of 61.2% (+19.5%), while maintaining lower parameter count than linear probing (41.7%). Similarly, in Figure 9e for the CUB200 dataset, our EP64 with _Do_ = _Do/_ 4 variant achieves comparable accuracy (75.9%) with computationally costly poolings such as SigLIP (77.8%) with around 7M trainable parameters less. 

Finally, in Figure 9f, GAP and [CLS], the two primary baselines, exhibit high parameter efficiency but low classification accuracy. In contrast, methods like SigLIP (54.39%), ViT (54.84%), and V- JEPA (52.44%) achieve higher accuracy, albeit at the cost of increasing the number of trainable parameters. EP has the best trade-off between accuracy and parameters, achieving top-1 classification accuracy of 53.7% with just 1M extra trainable parameters (EP64). 

Moving to the additional datasets shown in Figure 10, we observe consistent benefits of EP. On CIFAR-100 (Figure 10a), EP128 achieves 78.9%, which is close to the best-performing attention pooling methods, while using significantly fewer parameters. On Cars196 (Figure 10b), our EP64 variant achieves 82.7% top-1 accuracy, clearly surpassing repurposed pooling methods such as DELF, while requiring far fewer parameters. Smaller variants (e.g., EP16 and EP24) already provide substantial gains over linear probing, showing that even lightweight configurations of our method maintain strong performance on fine-grained datasets. 

20 

Published as a conference paper at ICLR 2026 



<!-- Start of picture text -->
(a) MAE ViT-S ImageNet-1k (b) MAE ViT-L ImageNet-1k<br>70 ViT EP16 EPEP1288 AIM CLIP ViT SigLIP<br>V-JEPA EPEP1664 EP4 EP4 CAE<br>65 EPEP648 AIM CLIP CaiT SigLIP 78 EP EP28 AbMILP EP2 DELF CoCaSimPool V-JEPACaiT<br>EPEP1696 EP4 CAE CoCa EPEP832<br>60 EP 4 EP2<br>55 EP EP EP16448 EP2AbMILP DELF 76 [CLS]<br>EP2 SimPool Do = Di Do = Di<br>Do = Di/ 2 Do = Di/ 2<br>50 EP EP 824 [CLS] Do = Di/ 4 74 Do = Di/ 4<br>EP2 GAP<br>GAP CBAM Do = Di/ 8 Do = Di/ 8<br>45<br>10 5 10 6 10 6 10 7<br>(c) SimMIM ViT-B ImageNet-1k (d) MAE VIT-B FGVC-Aircraft<br>ViT SigLIP<br>70<br>65 EP64 EPEP648 AIM CAE SigLIP CaiTV-JEPA EP24 EPEP324 EPEP464 CoCaAIM CAEV-JEPACaiTViT<br>EP64 EP8 EP4 CLIP 60 EP 4 DELF CLIP<br>60 EP 4 EP 2 CoCa AbMILP<br>EP8 SimPool<br>EP2<br>DELF<br>55 EP 48 EP2 GAP 50<br>EPEP28 [CLS] AbMILP SimPool Do = Di<br>[CLS]<br>50 Do = Di/ 2 CBAM Do = Di<br>Do = Di/ 4 40 Do = Di/ 4<br>CBAM Do = Di/ 8 GAP Do = Di/ 16<br>45<br>10 6 10 7 10 5 10 6 10 7<br>(e) MAE VIT-B CUB200 (f) MAE VIT-B Places365<br>80 EPEP864 EP64CoCa CAE SigLIP / CaiT ViT ViT<br>EPEP816 EP4 DELFEP4 AIM CLIP V-JEPA 54 EPEP8 64 AIM CAE SigLIP<br>70 AbMILP SimPool CaiT<br>EP4 CLIP<br>52 CoCa V-JEPA<br>60<br>DELF<br>[CLS] 50 SimPool<br>50 AbMILP<br>Do = Di<br>GAPCBAM Do = Di/ 4 48 [CLS] CBAM<br>40 D o = D i / 16 Do = Di<br>GAP<br>10 5 10 6 10 7 10 6 10 7<br>number of parameters number of parameters<br>top-1 accuracy (%)<br>top-1 accuracy (%)<br>top-1 accuracy (%)<br><!-- End of picture text -->

Figure 9: _Top-1 classification accuracy vs. number of trainable parameters_ (including the classifier) for two self-supervised learning methods, with backbones of varying size (a, b) and across various datasets (d, e, f). EP variants are marked with different colors for different output dimensionalities _Do_ . EP _M_ : efficient probing with _M_ learnable queries. [CLS]: linear probing using the class token; GAP: global average pooling over patch tokens; VIT: default transformer block. 

**Layer-wise probing.** Figure 11 presents a layer-wise comparison between standard linear probing (LP) and efficient probing (EP) using patch token representations from intermediate layers of a pre-trained and frozen MAE with ViT-B. While LP exhibits a clear degradation in performance as we move toward earlier layers (dropping from 67.7% at layer 12 to just 45.8% at layer 6), EP demonstrates remarkable robustness. It maintains high accuracy even from lower layers, with performance stabi- 

Figure 11: _ImageNet-1K accuracy from intermediate MAE ViT-B layers_ for LP vs. EP, with EP–LP gains. 

|LAYER|LP|EP|GAIN|
|---|---|---|---|
|12|**67.7**|75.6|+7.9|
|10|66.2|**75.9**|+9.7|
|9|64.5|75.4|+10.9|
|6|45.8|69.6|**+23.8**|



21 

Published as a conference paper at ICLR 2026 



<!-- Start of picture text -->
(a) MAE ViT-B CIFAR-100 (b) MAE ViT-B Cars-196<br>8078 EP EP EP4EP8 6496 EPEPEP16EP48 128 AIMCoCaCLIPCAE SigLIP V-Jepa ViTCaiT 80 EPEPEP81624 AbMILPEPEPEP164 64 EPEPEP16644 DELFAIMCoCaCAECLIP V-JEPA SigLIPCaiTViT<br>76 EP2 DELF 70 EP4 SimPool<br>SimPool<br>74 AbMILP<br>EPEP816 60<br>72 EP2<br>Do = Di Do = Di<br>GAP<br>70 [CLS] CBAM D o = D i / 4 50 CBAM D o = D i / 4<br>Do = Di/ 16 [CLS] GAP Do = Di/ 16<br>68<br>10 5 10 6 10 7 10 5 10 6 10 7<br>number of parameters number of parameters<br>top-1 accuracy (%)<br><!-- End of picture text -->

Figure 10: _Top-1 classification accuracy vs. number of parameters_ for MAE ViT-B on two datasets (a, b). We evaluate both dedicated probing mechanisms (e.g., V-JEPA) and repurposed attentive pooling methods (e.g., CLIP). EP variants are marked with different colors for different output dimensionalities _Do_ . EP _M_ : efficient probing with _M_ learnable queries. [CLS]: linear probing using the classification token; GAP: global average pooling over patch tokens; VIT: default transformer block. 

lizing beyond layer 9. Notably, EP yields a significant relative improvement of +23.8% at layer 6 over LP, underscoring its ability to extract and utilize meaningful representations from less semantically enriched stages of the encoder. These results highlight the effectiveness of EP in unlocking information from earlier layers that standard LP fails to exploit. 

**Low-shot probing.** Table 4 evaluates the performance of LP, EP, and FT under limited supervision, using only 5% and 10% of the ImageNet1K training set, stratified by class. Although LP struggles in this low-shot regime, EP substantially bridges the gap toward FT. Specifically, EP closes 74.8% and 71.5% of the LP _→_ FT performance gap for the 5% and 10% subsets, respectively. These improvements are particularly impressive given that EP remains significantly more parameter-efficient than FT, with a complexity comparable to that of LP. These findings highlight the strong data efficiency of EP. 

Table 4: _Top-1 accuracy on ImageNet-1k with limited training data for MAE ViT-B_ . Results for linear probing (LP), efficient probing (EP), and fine-tuning (FT) on 5% and 10% subsets. The last column shows the percentage of the LP _→_ FT performance gap closed by EP. For reference, the gap closed by EP on the full training set (100%) is 49.7%. 

|SUBSET|LP|EP|FT|% GAP|
|---|---|---|---|---|
|5%|49.6|60.9|64.7|74.8%|
|10%|55.9|65.2|68.9|71.5%|



**In- and out-of-domain k-NN evaluation.** To further examine how EP behaves relative to LoRAbased fine-tuning, we perform a cross-dataset k-NN evaluation on MAE ViT-B features, using ImageNet-1K, StanfordCars, and Food101 as target datasets (Table 5). The first row reports the baseline k-NN accuracy obtained directly from the frozen MAE backbone. The next three rows evaluate features produced by EP when the probe is trained on each dataset independently; the diagonal entries (e.g., 70 _._ 5% on ImageNet-1K, 70 _._ 0% on StanfordCars, 75 _._ 2% on Food101) correspond to in-domain performance, while off-diagonal entries measure cross-dataset generalization. The last three rows report the corresponding results when MAE is adapted with the best-performing LoRA configuration for each dataset. As expected, LoRA generally achieves the strongest in-domain accuracy (e.g., 72 _._ 3% on ImageNet-1K, 75 _._ 4% on StanfordCars, 80 _._ 3% on Food101), reflecting the benefit of supervised feature adaptation. However, EP consistently provides stronger or comparable out-of-domain performance: for instance, when trained on StanfordCars, EP features achieve 51 _._ 7% on ImageNet-1K and 42 _._ 0% on Food101, compared to 45 _._ 8% and 38 _._ 7% for LoRA-tuned features; similarly, EP trained on Food101 yields 58 _._ 2% on ImageNet-1K and 23 _._ 4% on StanfordCars, versus 56 _._ 7% and 15 _._ 2% for LoRA. These results suggest a complementary behavior: LoRA excels 

22 

Published as a conference paper at ICLR 2026 

at specializing the backbone to a specific task, while EP preserves more of the original pre-trained structure and thus offers more robust cross-dataset generalization. 

Table 5: _Cross-dataset k-NN evaluation on MAE ViT-B using frozen, EP-probed, and LoRA-tuned features_ . Default EP corresponds to EP32, while the “best LoRA” configuration applies LoRA to all 12 layers on the _WQ, WK, WV_ , and _WO_ projection matrices with rank _ρ_ =8. 

|FEATURES|K-N|N EVALUATION ON||
|---|---|---|---|
||IMAGENET-1K|STANFORDCARS|FOOD101|
|frozen MAE backbone|46.1|9.5|28.8|
|+ EP probed on ImageNet-1K|70.5|**31.4**|64.2|
|+ EP probed on StanfordCars|**51.7**|70.0|**42.0**|
|+ EP probed on Food-101|**58.2**|**23.4**|75.2|
|+ best LoRA tuned on ImageNet-1K|**72.3**|24.7|**65.6**|
|+ best LoRA tuned on StanfordCars|45.8|**75.4**|38.7|
|+ best LoRA tuned on Food-101|56.7|15.2|**80.3**|



**Object localization.** A key empirical property of EP is that its queries specialize in _complementary_ and semantically meaningful object parts (subsection 4.3). To examine EP utility beyond standard classification, we evaluate its performance in an unsupervised object localization setting, on ImageNet-1K (Table 6), using the WSOL protocol (Choe et al., 2020; 2022). We average EP ’s attention maps, without any modification or additional training, and compare them to the standard last-layer [CLS] _→_ patch attention. Across five backbones and three model sizes, EP consistently improves MaxBoxAccV2 by +9 _._ 8% on average. This demonstrates that EP ’s attention maps act as strong unsupervised localizers “outof-the-box”. 

Table 6: _Unsupervised ImageNet-1K localization_ (MaxBoxAccV2). EP substantially improves localization quality across five backbones and three model sizes, with an average gain of +9 _._ 8% over baseline attention. 

|MODEL|ARCH|LP|EP|∆ACC. (%)|
|---|---|---|---|---|
|MAE|ViT-S/16|46.0|58.5|+12.5|
|MAE|ViT-B/16|54.2|60.4|+6.2|
|MAE|ViT-L/16|46.9|61.2|+14.3|
|BEiTv2|ViT-B/16|47.0|61.2|+14.2|
|SimMIM|ViT-B/16|45.2|60.0|+14.8|
|iBOT|ViT-B/16|57.6|63.7|+6.1|
|SigLIP|ViT-L/16|44.0|44.2|+0.2|
|AVERAGE||48.7|**58.5**|+9.8|



**Image retrieval.** Table 7 and Table 8 evaluate EP in a zero-shot image retrieval setting, following the standard Recall@K protocol on two fine-grained datasets, CUB200 and Cars196, respectively. In both cases, we use the same EP mechanism trained once on ImageNet-1K, without any further adaptation on the target dataset. Across all retrieval experiments, we utilize the entire datasets (full set of images and classes), all of which remain fully unseen during the training stage on ImageNet1K. Every image serves as a query, ensuring a robust zero-shot evaluation protocol. On CUB200, EP consistently improves performance across five models (e.g., +40.8% for MAE, +11.2% for BEiTv2, +10.3% for iBOT, +5.3% for CLIP, +15.9% for SigLIP on R@1 metric). Similarly, on Cars196 the gains remain significant in most cases (e.g., +21.5% for MAE, +8.6% for iBOT, +4.6% for SigLIP). These findings show that EP ’s features can be used “out-of-the-box” for image retrieval, consistently outperforming the features of the frozen backbones. 

### C.3 EXPERIMENTAL ANALYSIS 

**Attention complementarity.** To further expand our analysis of attention complementarity, we complement the average-based metric reported in Figure 7 with an additional measure: 1 _−_ max offdiagonal similarity, which reflects the strongest redundancy between predictors. As shown in Figure 12a and Figure 12b, EP achieves substantially higher complementarity than the internal MHSA heads across all backbones, regardless of whether we use the average or max metric. A natural concern is whether this effect is biased by considering only the last block. To address this, we compute complementarity for MAE ViT-B across all 12 blocks (Figure 12c). The best scores obtained internally (0.37 for average, 0.10 for max) remain far below those of EP (0.65 and 0.22, respectively), confirming that the gap is not specific to the last block. Another question is whether the low diversity arises from self-attention itself. However, even SigLIP—which uses a cross-attention mechanism in 

23 

Published as a conference paper at ICLR 2026 

Table 7: _Zero-shot image retrieval performance (Recall@K) on CUB200 dataset (images: 11788, classes: 200)_ across five different backbones with and without Efficient Probing (EP). 

|FEATURES|R@1|R@2|R@4|R@8|R@16|R@32|
|---|---|---|---|---|---|---|
|frozen MAE backbone|16.3|23.7|33.3|45.0|58.0|70.8|
|+ EP on ImageNet-1K|57.1|69.5|79.5|87.5|93.0|96.2|
|frozen BEiTv2 backbone|57.0|68.2|78.0|85.8|90.9|94.2|
|+ EP on ImageNet-1K|68.2|78.7|86.7|92.3|95.9|97.8|
|frozen iBOT backbone|51.8|64.0|74.7|83.2|90.0|94.1|
|+ EP on ImageNet-1K|62.1|73.7|82.4|89.7|94.0|96.8|
|frozen CLIP backbone|75.0|84.3|91.3|95.3|97.5|98.7|
|+ EP on ImageNet-1K|80.3|88.0|92.9|95.8|97.6|98.6|
|frozen SigLIP backbone|60.8|72.1|81.6|88.8|93.5|96.4|
|+ EP on ImageNet-1K|76.7|85.7|92.0|95.7|97.8|98.8|



Table 8: _Zero-shot image retrieval performance (Recall@K) on Cars196 dataset (images: 16185, classes: 196)_ across five different backbones with and without Efficient Probing (EP). 

|FEATURES|R@1|R@2|R@4|R@8|R@16|R@32|
|---|---|---|---|---|---|---|
|frozen MAE backbone|12.7|17.6|23.7|31.8|42.9|55.7|
|+ EP on ImageNet-1K|34.2|44.8|55.6|65.9|76.4|85.2|
|frozen BEiTv2 backbone|48.3|60.7|72.3|81.8|89.2|94.5|
|+ EP on ImageNet-1K|44.2|56.4|68.5|79.1|87.3|93.7|
|frozen iBOT backbone|31.5|40.7|50.2|60.0|69.9|79.0|
|+ EP on ImageNet-1K|40.1|50.6|61.0|70.7|79.5|87.6|
|frozen CLIP backbone|79.6|88.9|94.6|97.8|99.2|99.7|
|+ EP on ImageNet-1K|79.8|89.3|94.8|97.8|99.2|99.7|
|frozen SigLIP backbone|85.8|92.3|96.1|98.0|99.1|99.6|
|+ EP on ImageNet-1K|90.4|95.4|97.8|99.0|99.6|99.8|



its last block—still yields lower complementarity than EP, suggesting that the effect is not explained by self-attention alone. 

Figure 12d and Figure 12e further compare EP against other attentive probing methods (V-JEPA, AIM). Interestingly, all attentive probing mechanisms achieve higher complementarity than the internal MHSA heads, with AIM coming close to EP. This pattern might suggest that attentive probing, when designed effectively, encourages predictors to specialize in complementary regions, likely because probing operates on frozen backbones and must learn to aggregate features as efficiently as possible. This evidence indicates that complementarity may be an inherent property of attentive probing, rather than a byproduct of backbone architecture or attention type, opening new directions for future work. 

**Attention complementarity as a loss.** We further examine the role of query complementarity by encouraging diversity between queries/heads via an auxiliary attentionsimilarity loss. Specifically, for AIM, V-JEPA, and EP we add to the cross-entropy loss, an extra term _L_ attn that penalizes similarity between the attention maps of different queries. On MAE ViT-B probed on ImageNet-1K, this yields a small gain for V-JEPA (74 _._ 1% _→_ 74 _._ 3%), while AIM and EP retain essentially identical final accuracy but converge faster (e.g. EP reaches 64 _._ 1% at epoch 10 with _L_ attn vs. 60 _._ 0% without it, before both converge to the same final score). These results suggest that complementarity is a beneficial property that can modestly help or accelerate training. However, methods such as AIM and EP, learn highly 



<!-- Start of picture text -->
60<br>40<br>EP with attn-sim<br>20 EP<br>0 20 40 60 80<br>epoch<br>Figure 13: Validation top-1 accuracy on<br>ImageNet-1K over training epochs for EP, with<br>and without the auxiliary attention-similarity loss.<br>The additional loss slightly accelerates early train-<br>ing but both variants converge to identical final<br>performance.<br>validation accuracy (%)<br><!-- End of picture text -->

24 

Published as a conference paper at ICLR 2026 



<!-- Start of picture text -->
0 . 84<br>0 . 8 0 . 75 0 . 75 0 . 76 MHSA 0 . 75 0 . 75<br>0 . 65 0 . 69 EP<br>0 . 61<br>0 . 6<br>0 . 44 0 . 47 0 . 46<br>0 . 4<br>0 . 24 0 . 27 0 . 24 0 . 27 0 . 22<br>0 . 2 0 . 13 0 . 18 0 . 16<br>0<br>MAE MAE iBOT DINOv2 DINOv2 DINOv3 DINOv3 Franca CLIP SigLIP<br>ViT-B ViT-L ViT-B ViT-B ViT-L ViT-B ViT-L ViT-L ViT-L ViT-L<br>(a) Complementarity measured as 1 minus average off-diagonal similarity.<br>0 . 3<br>MHSA<br>0 . 22 EP 0 . 23<br>0 . 2 0 . 2 0 . 18 0 . 19 0 . 17 0 . 18 0 . 18<br>0 . 15<br>0 . 13<br>0 . 1 0 . 09 0 . 07 0 . 05 0 . 06<br>0 . 02<br>0 . 004 0 . 006 0 . 004 0 . 004 0 . 003<br>0<br>MAE MAE iBOT DINOv2 DINOv2 DINOv3 DINOv3 Franca CLIP SigLIP<br>ViT-B ViT-L ViT-B ViT-B ViT-L ViT-B ViT-L ViT-L ViT-L ViT-L<br>(b) Complementarity measured as 1 minus max off-diagonal similarity.<br>0 . 4 0 . 37 average<br>max<br>0 . 29 0 . 29<br>0 . 3 0 . 26 0 . 23 0 . 23 0 . 24 0 . 25 0 . 27 0 . 24<br>0 . 2 0 . 19<br>0 . 14<br>0 . 1 0 . 09 0 . 09<br>0 . 1 0 . 07 0 . 06 0 . 07 0 . 06 0 . 06<br>0 . 04 0 . 04<br>0 . 02 0 . 02<br>0<br>1 2 3 4 5 6 7 8 9 10 11 12<br>(c) Complementarity of MAE ViT-B across all 12 blocks, using both average and max.<br>00 .. 86 0 . 55 0 . 59 0 . 65 0 . 63 0 . 76 0 . 77 0 . 52 0 . 65 0 . 67 0 . 61 0 . 7 0 . 71 MHSA<br>V-JEPA<br>0 . 4 0 . 36 0 . 37 0 . 33 AIM<br>0 . 24 EP<br>0 . 2<br>0<br>MAE BEiTv2 SimMIM CAPI<br>ViT-B ViT-B ViT-B ViT-L<br>(d) Complementarity measured as 1 minus average off-diagonal similarity.<br>0 . 3<br>0 . 25<br>0 . 22 0 . 23 0 . 22 0 . 23<br>0 . 2 0 . 17 0 . 2 0 . 17 0 . 19 MHSA<br>V-JEPA<br>0 . 14 0 . 13<br>0 . 1 0 . 09 0 . 08 0 . 11 0 . 1 AIMEP<br>0 . 04<br>0<br>MAE BEiTv2 SimMIM CAPI<br>ViT-B ViT-B ViT-B ViT-L<br>(e) Complementarity measured as 1 minus max off-diagonal similarity.<br>Complementarity<br>Complementarity<br>Complementarity<br>Complementarity<br>Complementarity<br><!-- End of picture text -->

Figure 12: _Complementarity scores of attention maps across different backbones (a, b, c) and probing methods (d, e)._ We compare the diversity of internal MHSA heads in the last block against the external V-JEPA heads, AIM heads, and EP queries. 

25 

Published as a conference paper at ICLR 2026 

complementary queries, without _L_ attn, via the attention mechanism that naturally discovers this structure on its own. 

**Entropy.** Figure 14 examines the relation between last block attention entropy and probing performance across different pre-training methods. Models with lower entropy exhibit more concentrated and focused attention distributions, correlating with stronger probing accuracy under EP. This is particularly evident for methods like as DINOv2 and DINOv3, which couple low entropy with strong LP and EP performance. In contrast, models like MAE ViT-S (MAE-S) or SimMIM ViT-B (SimMIM-B) show higher entropy, reflecting more diffuse attention and correspondingly weaker probing under LP. Crucially, EP consistently boosts accuracy across all entropy levels, with bubble sizes indicating particularly large gains for the high-entropy models. This aligns with our broader finding that methods optimizing patch-level representations rather than explicit global tokens benefit most from attentive probing, as EP effectively compensates for diffuse attention and makes probing more robust to the quality of backbone distributions. 



<!-- Start of picture text -->
EP<br>90<br>DINOv3-L<br>Franca-L DINOv3-B DINOv2-B<br>DINOv2-L 85<br>80 CLIP-L CAPI-L BEITv2-B iBOT-B MAE-L<br>DINO-B<br>80<br>MAE-B<br>70<br>75<br>60<br>Simmim-B 70<br>50<br>MAE-S<br>65<br>40<br>3 3 . 2 3 . 4 3 . 6 3 . 8 4 4 . 2 4 . 4 4 . 6 4 . 8 5<br>[CLS]  → patch tokens entropy (last block)<br>LP<br><!-- End of picture text -->

Figure 14: _Entropy analysis of last-block attention distribution across different pre-training methods._ Bubble color indicates efficient probing (EP) accuracy, bubble size encodes ∆ accuracy (EP _−_ LP). Lower entropy corresponds to more focused attention and higher accuracy under EP. 

**Impact of attention predictors,** _Da_ **, and** 

_Do_ **.** We analyze the effect of increasing the number _M_ of heads (in AIM) and the number _M_ of queries (in EP) on probing performance. Figure 15 shows that both lead to accuracy improvements. For AIM, increasing the number of heads incurs no additional cost in terms of parameters, but its effectiveness depends on the presence of _WK_ . In contrast, EP achieves similar or better performance by leveraging additional queries, while removing _WK_ . AIM introduces an additional attention dimensionality _Da_ , since its query is learnable and interacts with _WK_ . Lowering _Da_ reduces the parameters but leads to a greater accuracy drop (green points), indicating that the learned query formulation benefits from a large attention space. We also evaluate the impact of reducing the output dimensionality _Do_ (blue points). On EP, we observe that lowering _Do_ to _Di/_ 2 reduces 



<!-- Start of picture text -->
EP64 AIM64<br>AIM64<br>EP16<br>AIM16<br>75 AIM 16<br>EP64<br>AIM64 AIM64 AIM4 AIM4<br>EP4<br>EP16<br>74 AIM16<br>EP4 AIM16 AIM4<br>EP2<br>AIM4 AIM2 AIM2<br>73 EP2<br>AIM2 Do = Di<br>AIM2 Do = Di/ 2<br>Da = Di/ 2<br>72<br>1  ·  10 6 1 . 5  ·  10 6 2  ·  10 6<br>number of parameters<br>top-1 accuracy (%)<br><!-- End of picture text -->

Figure 15: _Effect of varying the number of heads/queries M , Da, and output dimension Do_ on probing accuracy. Black: standard ( _Do_ = _Di_ ); blue: reduced classifier dimension ( _Do_ = _Di/_ 2); green: reduced attention dimension ( _Da_ = _Di/_ 2); half-bluehalf-green: simultaneous reduction of both _Do_ and _Da_ . 

26 

Published as a conference paper at ICLR 2026 

parameters while maintaining competitive performance. Interestingly, this strategy also generalizes well to AIM, demonstrating that extracting lower-dimensional features can achieve comparable accuracy with reduced computational cost. 

**EP convergence.** Although all probing methods are trained for 90 epochs for fairness (following the protocol of He et al. (2022) and Przewike´zlikowski et al. (2025)), we also examine how quickly EP converges. As shown in Table 9, EP trained for only 10 epochs (EP@10) already matches or even surpasses the performance of standard linear probing trained for the full 90 epochs (LP@90) across 12 models. Moreover, EP recovers on average more than 97% of its final accuracy within these first 10 epochs. These results highlight that in practical scenarios, EP requires only a small fraction of the standard training budget to achieve near-optimal probing performance and produce highly discriminative features. 

Table 9: _EP convergence on ImageNet-1K._ EP@10 is EP accuracy after 10 epochs; LP@90 and EP@90 are final probing results after 90 epochs. 

|MODEL|EP@10|LP@90|EP@90|EP@10 VS. LP@90|GAP|EP@10 VS. EP@90 GAP|
|---|---|---|---|---|---|---|
|MAE ViT-B/16|60.0|67.7|75.6||88.6%|79.4%|
|BEiTv2 ViT-B/16|80.0|79.0|81.7||101.3%|97.9%|
|CAPI ViT-L/14|82.3|81.5|83.6||101.0%|98.4%|
|iBOT ViT-B/16|78.6|78.7|79.2||99.9%|99.2%|
|DINOv2 ViT-B/14|83.6|83.2|84.0||100.5%|99.5%|
|DINOv2 ViT-L/14|85.0|85.2|85.6||99.8%|99.3%|
|Franca ViT-B/16|83.8|83.8|84.3||100.0%|99.4%|
|DINOv3 ViT-B/16|84.4|84.0|84.4||100.5%|100.0%|
|DINOv3 ViT-L/14|86.7|86.6|87.1||100.1%|99.5%|
|CLIP ViT-L/14|82.7|82.3|83.4||100.5%|99.2%|
|SigLIP ViT-L/16|85.6|84.1|86.1||101.8%|99.4%|
|AIMv2 ViT-L/14|85.5|84.8|85.9||100.8%|99.5%|



**Attention mixing.** We further explore an attention mixing variant of EP, where a larger set of query maps is linearly projected into a smaller number of effective maps. For example, projecting 64 queries into 8 maps achieves 75.6% accuracy, outperforming the 8- query baseline (74.8%) and suggesting that mixing can improve the expressivity of a small query budget. However, compared to the unprojected 32 or 64 query variants (75.6%), attention mixing offers no clear advantage while introducing additional parameters through the projection layer. For this reason, we do not include attention mixing in the final design of EP, though it may inspire future extensions exploring alternative aggregation strategies. 

Table 10: _Comparison of_ EP _variants using different numbers of original maps (queries) and projected maps on ImageNet-1K with MAE ViT-B._ Attention mixing refers to projecting multiple query maps into a smaller set via a linear+softmax layer. While projection improves over directly training with few queries (e.g., 64→8 vs. 8 original), it does not outperform the un-projected setting with the same number of queries, and adds extra parameters. 

|# ORIGINAL <br>EP MAPS|# PROJECTED<br>EP MAPS|ACCURACY|# ORIGINAL <br>EP MAPS|ACCURACY|
|---|---|---|---|---|
|32|8|75.4|||
|64|8|**75.6**|8|74.8|
|128|8|75.5|||
|64|16|75.5|||
|128|16|**75.8**|16|75.0|
|256|16|75.6|||
|64|32|75.6|||
|128|32|75.6|32|75.6|
|256|32|**75.7**|||
|128|64|75.8|||
|256|64|75.8|64|75.6|
|384|64|**75.9**|||
|256|128|75.8|128|755|
|384|128|**76.0**||.|



**Matryoshka representation learning.** In our accuracy vs. parameter trade-offs (e.g., Figure 2), each evaluation scale ( _Do_ = _Di, Di/_ 2 _, Di/_ 4 _, Di/_ 8) requires training a separate classifier, and the resulting accuracies potentially serve as an _upper bound_ for that dimensionality, since each probe is fully specialized to its target output size. To reduce training cost and enable a single probe that operates across multiple scales, we investigate whether _Matryoshka representation learning_ (Kusupati 

27 

Published as a conference paper at ICLR 2026 

et al., 2022)—originally proposed for training from scratch or fine-tuning—can be applied to attentive probing under EP. Matryoshka jointly optimizes probing across multiple output dimensionalities by summing their losses (e.g., _L_ = _λLD_ + _LD/_ 2). We study two variants: (i) Efficient Matryoshka, which uses a single classifier with nested subspaces spanning multiple dimensionalities, and thus adds no extra parameters; (ii) Vanilla Matryoshka, which trains separate classifiers for each dimensionality, increasing the parameter count. Table 11 reports results with MAE ViT-B probed using EP. Without Matryoshka, probing at _D_ ( _Do_ = _Di_ ) achieves the highest accuracy (75.6%), but performance drops sharply at smaller dimensions (50.7% at _D/_ 2, 20.1% at _D/_ 4, 5.5% at _D/_ 8). Efficient Matryoshka substantially improves performance at reduced dimensions (e.g., +22.5% at _D/_ 2) while maintaining accuracy at _D_ . Accuracy at _D_ slightly decreases as _λ_ (the weight of the full-dimensional loss) is reduced, reflecting the trade-off between full- and low-dimensional performance. Vanilla Matryoshka alleviates the performance drop at _D_ but requires more parameters. To further illustrate these effects, we evaluate the standard linear probing (LP) baseline without Matryoshka under the same _D/_ 2, _D/_ 4, _D/_ 8 evaluation protocol, as also extending LP with Matryoshka. In Table 12 we observe that standard LP (without Matryoshka) suffers from an even sharper performance collapse than EP (e.g., 34.5% at _D/_ 2, 10.8% at _D/_ 4, and 1.4% at _D/_ 8). This confirms that the drop at reduced dimensions is not specific to EP, but is an inherent limitation of evaluating ViT representations at truncated dimensionalities without explicit multi-scale optimization. Overall, Matryoshka probing provides a promising way to probe across multiple scales simultaneously, improving efficiency, and adaptability of evaluation. However, it does not yet match the upper-bound performance of dimension-specific probes, thus we present it here as an exploratory analysis that highlights an exciting direction for future work. 

Table 11: _Comparison of_ EP _with and without Matryoshka representation learning_ on ImageNet1K (MAE ViT-B). Without Matryoshka, probing at _D_ ( _Do_ = _Di_ ) achieves the best accuracy but collapses at smaller dimensions. Efficient Matryoshka enables multi-scale probing without extra parameters, improving performance at reduced dimensions with relatively small degradation at _D_ . Vanilla Matryoshka partly restores performance at _D_ but at the cost of additional parameters. 

||WIT|HOUTM|ATRYOSH|KA|E|FFICIENTM|ATRYOSH|KA|VANILLAM|ATRYOSHKA|
|---|---|---|---|---|---|---|---|---|---|---|
||_D_|_D/_2|_D/_4|_D/_8||_D_,_D/_2,|_D/_4,_D/_8||_D_,_D/_2,|_D/_4,_D/_8|
|EVAL. ON|||||_λ_=0_._8|_λ_=0_._6|_λ_=0_._4|_λ_=0_._25|_λ_=0_._6|_λ_=0_._25|
|_D_|**75.6**|–|–|–|75.0|74.7|74.3|73.9|75.4|75.2|
|_D/_2|50.7|**74.4**|–|–|72.2|73.1|73.6|73.8|73.8|74.4|
|_D/_4|20.1|–|**72.4**|–|69.5|70.9|71.4|71.6|71.7|72.2|
|_D/_8|5.5|–|–|**69.8**|65.7|67.3|67.9|68.2|68.6|68.9|
|# PAR.|1.4M|1.4M<br>5.5|1.4M<br>M|1.4M||1.|4M||2.1|M|



Table 12: _Comparison of LP with and without Matryoshka representation learning_ on ImageNet-1K (MAE ViT-B). Without Matryoshka, linear probing collapses sharply at reduced output dimensionalities. Matryoshka improves low-dimensional performance, without increasing parameters, though the accuracy at _D_ decreases as _λ_ (the weight of the full-dimensional loss) is reduced. 

|EVAL. ON|WITHOUTMATRYOSHKA||EFFICIENTM|ATRYOSHK|A|
|---|---|---|---|---|---|
|||_λ_ = 0_._8|_λ_ = 0_._6|_λ_ = 0_._4|_λ_ = 0_._25|
|_D_|**66.4**|66.1|65.8|65.4|65.0|
|_D/_2|34.5|59.2|60.6|61.2|61.4|
|_D/_4|10.8|52.5|54.5|55.3|55.7|
|_D/_8|1.4|43.4|46.1|47.0|47.4|



**Visualizations.** To better understand the behavior of different attentive pooling/probing methods, we present visualizations of attention maps across various configurations. 

Figure 16 explores the effect of varying the number of queries in EP, visualizing configurations with 1, 2, 4, and more queries. When only a single query (EP1) is used, the attention map tends to capture a coarse, global representation of the object. As the number of queries increases, the attention becomes more fine-grained and spatially distributed, with each query specializing in distinct object 

28 

Published as a conference paper at ICLR 2026 



<!-- Start of picture text -->
original EP1 EP2 EP2 EP4 EP4 EP4 EP4<br>image q 1 q 1 q 2 q 1 q 2 q 3 q 4<br>EP8 EP8 EP8 EP8 EP8 EP8 EP8 EP8<br>q 1 q 2 q 3 q 4 q 5 q 6 q 7 q 8<br>EP16 EP16 EP16 EP16 EP16 EP16 EP16 EP16<br>q 1 q 2 q 3 q 4 q 5 q 6 q 7 q 8<br>EP16 EP16 EP16 EP16 EP16 EP16 EP16 EP16<br>q 9 q 10 q 11 q 12 q 13 q 14 q 15 q 16<br><!-- End of picture text -->



Figure 16: _Attention maps of efficient probing (_ EP _)_ variants grouped by the number of queries (1, 2, etc.). MAE ViT-B pre-trained on ImageNet-1K, probed with EP. Images: ImageNet-1k validation set. 

regions. This highlights the flexibility of EP in controlling the granularity of attention: fewer queries encourage holistic coverage, while more queries promote detailed, part-based localization. 

Figure 17 shows the attention maps obtained from four single-head attention probing methods (CBAM, AbMILP, DELF, and SimPool) using an ImageNet-1K pretrained MAE ViT-B model. Among them, CBAM exhibits poor localization, often failing to focus on the target object, which is consistent with its low classification accuracy across datasets. In contrast, AbMILP, DELF, and SimPool produce more precise and meaningful attention, highlighting relevant object regions while suppressing background noise. Due to their single-head nature, these methods are compelled to concentrate all semantic information into a single attention vector, which encourages a global view of the input image rather than fine-grained discrimination. 

Figure 18 compares attention maps from multi-head probing methods. Rather than visualizing just the average attention across heads,which can obscure useful per-head behavior, we show the minimum, maximum, and standard deviation across attention heads. The first column contains maps from the [CLS] token of the pretrained MAE ViT-B model. The remaining columns display maps from CAE, CaiT, CLIP, CoCa, ViT, V-JEPA, SigLIP, and AIM, alongside EP using 16 learnable queries (EP16). Notably, EP produces high-quality attention maps that rival the best-performing methods in both clarity and relevance, while retaining computational efficiency. 

Figure 19 presents the attention maps corresponding to each individual query in EP. We observe that each query **_q_** _i_ attends to distinct, complementary regions of the object (e.g., head, torso, boundaries), illustrating how EP distributes attention cooperatively across salient features without redundancy. This diversity among queries reveals the model’s capacity to decompose complex objects into meaningful sub-parts. 

29 

Published as a conference paper at ICLR 2026 







































































































<!-- Start of picture text -->
input CBAM AbMILP DELF SimPool<br>image<br><!-- End of picture text -->

Figure 17: _Attention maps of single-head attention pooling methods_ . MAE ViT-B pre-trained on ImageNet-1k. Images: ImageNet-1k validation set. 

30 

Published as a conference paper at ICLR 2026 



<!-- Start of picture text -->
[CLS] CAE CaiT CLIP CoCa ViT V-JEPA SigLIP AIM EP16<br>mean<br>std<br>min<br>max<br>mean<br>std<br>min<br>max<br><!-- End of picture text -->

Figure 18: _Attention maps of multi-head attention pooling methods_ for different attention predictor aggregators: mean, standard deviation (std), minimum (min), and maximum (max). MAE ViT-B pre-trained on ImageNet-1K. Images: ImageNet-1k validation set. EP16: efficient probing (EP) with 16 queries. 



































































<!-- Start of picture text -->
q 1 q 2 q 3 q 4 q 5 q 6 q 7 q 8<br><!-- End of picture text -->

Figure 19: _Attention maps of efficient probing (_ EP _)_ with 8 queries. Each query **_q_** _i_ learns to focus on distinct and complementary regions, capturing diverse spatial and semantic information. MAE ViT-B pre-trained on ImageNet-1K, probed with EP. Images: ImageNet-1k validation set. 

31 

Published as a conference paper at ICLR 2026 

## D LIMITATIONS AND FUTURE WORK 

While our study provides the first systematic benchmark of attentive probing and introduces a lightweight yet effective alternative, several limitations remain. First, our evaluation focuses exclusively on frozen backbones. Although this setting isolates the effect of probing mechanisms, it leaves open how efficient probing might interact with lightweight fine-tuning strategies or adapterbased methods. Exploring the synergy between probing and parameter-efficient fine-tuning could offer a broader view of scalable evaluation. 

Second, our current experimental protocol always performs a full forward pass of the backbone during training, even though the backbone is frozen. In practice, one could pre-compute and store patch-token features once and run probing purely on top of these cached features. Systematically studying this “frozen-feature” protocol—its memory/IO trade-offs, impact on optimization dynamics, and applicability across architectures—is an interesting direction for future work. 

Third, while we introduced complementarity metrics (average and max similarity) to study the diversity of attention predictors, our analysis remains largely diagnostic. A deeper theoretical understanding of why attentive probing tends to yield more complementary maps—and whether this property can be explicitly optimized—could further extend probing beyond evaluation into representation refinement. 

Fourth, our experiments concentrate on image classification benchmarks. Attentive probing may also benefit other tasks that naturally require part-level reasoning, such as detection, segmentation, or retrieval, where its complementary attention maps could act as implicit part detectors. Extending probing to such structured tasks is a promising direction. 

Fifth, although we examined variants such as attention mixing and Matryoshka probing, both were limited to ImageNet-scale experiments. Their potential on larger models and multimodal settings (e.g., vision–language tasks) remains underexplored. Similarly, the alternative protocol of probing with pre-stored frozen features, requiring only one backbone forward pass, is left for future work. 

Finally, the broader implications of probing as more than an evaluation protocol deserve attention. Our results suggest that attentive probing exhibits emerging properties—such as diversity and interpretability—that are not trivially inherited from the backbone. Understanding whether these properties generalize across modalities and can be exploited for tasks like explainability, robustness, or adaptive computation opens up an exciting line of future research. 

## E USE OF LARGE LANGUAGE MODELS 

This paper has made limited use of large language models (LLMs), specifically to aid in the polishing and refinement of writing. LLMs were not used for ideation, technical contributions, experimental design, analysis, or related work retrieval. All research ideas, methodology, experiments, and conclusions presented are solely the work of the authors. 

32 

