This CVPR Workshop paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore. 

# **Nexar Dashcam Collision Prediction Dataset and Challenge** 

Daniel C. Moura Nexar Inc. 

daniel.moura@getnexar.com 

Shizhan Zhu Nexar Inc. 

shizhan.zhu@getnexar.com 

## Orly Zvitia Nexar Inc. 

orly.zvitia@getnexar.com 

## **Abstract** 

_This paper presents the Nexar Dashcam Collision Prediction Dataset and Challenge, designed to support research in traffic event analysis, collision prediction, and autonomous vehicle safety. The dataset consists of 1,500 annotated video clips, each approximately 40 seconds long, capturing a diverse range of real-world traffic scenarios. Videos are labeled with event type (collision/near-collision vs. normal driving), environmental conditions (lighting conditions and weather), and scene type (urban, rural, highway, etc.). For collision and near-collision cases, additional temporal labels are provided, including the precise moment of the event and the alert time, marking when the collision first becomes predictable._ 

_To advance research on accident prediction, we introduce the Nexar Dashcam Collision Prediction Challenge, a public competition on top of this dataset. Participants are tasked with developing machine learning models that predict the likelihood of an imminent collision, given an input video. Model performance is evaluated using the average precision (AP) computed across multiple intervals before the accident (i.e. 500 ms, 1000 ms, and 1500 ms prior to the event), emphasizing the importance of early and reliable predictions._ 

_The dataset is released under an open license with restrictions on unethical use, ensuring responsible research and innovation._ 

## **1. Introduction** 

Traffic accidents cause significant loss of life and financial damage worldwide each year. Preventing various types of traffic accidents, such as vehicle collisions, vehiclepedestrian collisions, and single-vehicle loss of control, can save lives and protect assets. Research in autonomous driving and ADAS (Advanced Driver Assistance Systems)[9, 15] shows promising potential for improving driving safety. This progress is driven by the widespread adoption of dashcams, which provide a field of view comparable to or even more comprehensive than that of a human driver. Early an- 

ticipation of imminent traffic accidents[7] adds significant value to autonomous driving systems. These systems can continuously analyze dashcam footage in real-time, aiming to make timely and confident predictions about potential accidents affecting the vehicle or occuring nearby. 

Table 1. Summary of the existing datasets and their properties: total number of video clips and frames, availability of attributes annotations, resolution, S/R (synthetic or real), and if open-sourced (“A” indicates academic use only). MMAU [11, 12] also provides re-annotations the CCD [4], DoTA [54, 55] and A3D [56] datasets. 

||#Clips|#Frames|Attr.|Res.|S/R|Open<br>source|
|---|---|---|---|---|---|---|
|DAD [7]|1,750|175K|-|**720**_×_**1280**|**R**|A|
|SA [58]|1,733|173K|-|**720**_×_**1280**|**R**|-|
|EpicFail [58]|3,000|_≈_300K|-|-|**R**|A|
|VIENA<sup>2 </sup>[2]|**15K**|**2.25M**|-|**1280**_×_**1920**|S||
|NIDB [25, 47]|6,244|**1.3M**|-|-|**R**|-|
|A3D [56]|1,500|208K|<sup>1</sup>|**720**_×_**1280**|**R**|-<sup>2</sup>|
|CTA [57]|1,935|853K|-|**720**_×_**1280**|**R**||
|CCD [4]|4,500|225K||**720**_×_**1280**|**R**||
|DADA-2000 [13]|1,962|649K||**660**_×_**1584**|**R**||
|GTACrash [26, 27]|11,381|228K|-|400_×_710|S||
|YTCrash [26, 27]|222|4,440|-|400_×_710|**R**||
|ROL [24]|1,000|100K|-|**720**_×_**1080**|**R**||
|TRA [33]|2,000|40-200K|-|_≥_**720**_×_**1280**|**R**|A|
|DeepAccident [52]|-|57K||**900**_×_**1600**|S||
|DoTA [54, 55]|4,990|504K|<sup>3</sup>|**720**_×_**1280**|**R**||
|CTAD[38]|1,100|727K||480_×_640|S||
|**Nexar**|1,500|**1.70M**||**720**_×_**1280**|**R**||



A key challenge in accident prediction is its online nature. Unlike accident detection [21], where the entire video sequence is available offline, here the system needs to predict an event (accident) before it occurs. Poor judgment can result in failing to prevent or mitigate an accident (false negative), or taking unnecessary preventive actions (false positive) that may introduce new risks, such as sudden braking. Moreover, anticipation time is critical - detecting an accident too late, even if correct, may render the prediction ineffective. Therefore, both accuracy and anticipation time are crucial. This motivated us to propose an evaluation metric that integrates both aspects into a single score: the Average 

> 1Re-annotated by MMAU [12]. 

> 2The YouTube videos following the URLs are no longer available. 

> 3Re-annotated by MMAU [12]. 

2583 



<!-- Start of picture text -->
… …<br>t=546 t=585 t=604 t=624 t=663<br>… …<br>t=372 t=477 t=529 t=582 t=687<br>Before Accident Time of Alert Time of Accident After Accident<br><!-- End of picture text -->

Figure 1. Two representative samples from the Nexar dataset. First column represents **Before accident interval** , columns 2-4 represent **Alert interval** , and fifth column represent **After accident interval** . Within the alert interval, column 2 represents the “time-of-alert” which is the earliest moment that the driver could intervene to prevent the accident. column 4 represents the “time-of-accident”, and column 3 represents an intermediate frame between “time-of-alert” and “time-of-accident”. 

Precision over different times to the event. 

Another challenge in accident anticipation is the subjectivity of determining when an accident can be predicted and when preventive action should be taken. This uncertainty complicates dataset construction, leading to multiple temporal labeling definitions, each reflecting different perspectives on when an accident begins [12, 13, 24, 54–56]. We define the “time of alert” as the earliest moment when a focused human can recognize that a dangerous situation is about to occur. To mitigate subjectivity, we derive the “time of alert” from annotations provided by multiple annotators. This “time of alert” serves as a baseline for setting expectations on when a computer-based model should predict the accident. 

This paper presents the Nexar Dashcam Crash Prediction Challenge for evaluating early traffic accident anticipation, accompanied by a newly created dataset. Our temporal annotations for accidents and near-collisions divide videos into three intervals (Figure 1): 

1) **Before accident interval** , which starts at the beginning of the video and ends at the “time of alert,” the earliest moment a driver should intervene to prevent a possible accident. 

2) **Alert interval** , which starts at the “time of alert” and ends at the “time of accident/near accident,” the actual collision time point and the last moment when anticipation is meaningful. 

3) **After accident interval** , which begins after the collision and extends until the end of the video. 

Our dataset presents significant domain challenges compared to existing datasets. It includes a substantial portion of highly challenging videos where the alert interval lasts only a few frames, as illustrated in Figure 2. Additionally, it captures various lighting conditions, weather situations, and road types, along with camera-related artifacts such as reflections, motion blur, fog, and lens flare (Figure 3). 

The rest of the paper is organized as follows. Section 2 

reviews the related literature. We introduce the dataset and its annotation process in Section 3. Section 4 outlines the challenge details. We summarize our contributions in Section 5 and conclude by suggesting potential future directions in Section 6. 

## **2. Related Work** 

**Traffic Accident Anticipation** . The challenge presented in this paper concerns Traffic Accident Anticipation (TAA) [1, 4, 5, 7, 8, 11, 14, 17, 19, 20, 22–24, 27–32, 34–36, 39– 41, 43, 45–47, 50, 51, 58]. TAA differs from Traffic Accident Detection (TAD) [21] in that TAA requires early anticipation of an accident before observing future sequences, making it significantly more challenging than determining when an accident happens in a complete video sequence. Several efforts have been made to address the TAA task. Early works, including [19], prior to the deep learning era, anticipated potential traffic accidents using vehicle trajectory cues. Chan et al. [7] formally defined the task, introduced an evaluation metric, and proposed the first large-scale dataset for evaluation. This was followed by Zeng et al. [58] and Suzuki et al. [47], who introduced novel architectural designs and loss functions. Most approaches [4, 7, 27, 40, 47, 58] follow a framework where a feature extraction network captures essential spatial and temporal cues, followed by a recurrent unit that accumulates and propagates temporal information to generate predictions. Other approaches leverage region-based feature extraction followed by a graph convolutional network to model inter-relations, as proposed in [4, 40]. Bao et al. [5] also applied a reinforcement learning scheme to predict potential accident risks. With the rise of vision transformer architectures [49], which have proven effective in various vision tasks [6, 10, 37, 48, 53], self-attention mechanisms have also been employed for accident anticipation [23, 24, 46]. 

2584 

**Dashcam-based datasets for traffic accident anticipation** . To facilitate research and evaluation, various datasets have been introduced alongside the development of algorithms for TAA [2–4, 7, 12, 13, 18, 24–26, 33, 52, 55–58]. In general, datasets for TAA require both positive (accident occurred) and negative (no accident) video clips, in contrast to the positive-only nature of temporal localization tasks in TAD. Various datasets provide additional annotations, including spatial annotations [4], temporal annotations [12, 13], scene attribute annotations [4, 12, 13, 38], driver attention labels [13], accident categories [11], as well as language-based descriptions and reasoning [12]. Synthetic datasets [2, 26, 38, 52] have also been introduced, offering advantages such as large-scale data availability and fully annotated ground truth. Table 1 provides statistics and additional properties of these datasets. 

Comparing to existing datasets, our dataset provides significantly more frames per video, offering a dual advantage: a richer temporal context for accident anticipation and a more comprehensive feature space, enabling models to better distinguish between predictive cues that reliably indicate accidents and those that do not necessarily lead to a collision. 

**Temporal Annotations for Accident Anticipation** . Existing datasets explore different temporal annotations of accidents. Prior studies can be categorized into two main approaches based on their criteria for defining the start and end times of an event, as summarized in Table 2. 

1) **Appearance-based** [12, 13, 24], where the start time is defined as the initial full or partial appearance of the vehicle that eventually causes the accident ( _t_ appearing), and the end time is set as the collision moment ( _t_ collision). 

2) **Anomaly- or causality-based** [54–57], where the start time corresponds to the moment the annotator perceives the accident as inevitable or when the vehicle begins to exhibit abnormal behavior ( _t_ inevitable), while the end time is defined as the point when all vehicles are either out of sight or stationary ( _t_ end of anomaly<sup>).</sup> 

While 1) provides a more objective annotation, it is biased toward earlier timestamps since the mere appearance of a third-party vehicle does not necessarily indicate the moment when the accident becomes predictable. Conversely, 2) is subjective and often lacks precise temporal annotation for the exact collision moment. 

Although MMAU [12] has made significant efforts to reannotate several datasets and bridge the gap between different temporal annotation strategies, its annotations are only available for a subset of samples within each dataset and may still introduce bias when combined with the original temporal annotations. 

We believe our annotation approach offers the best combination of the advantages of both “anomaly-based” and “appearance-based” temporal definitions. Specifically, 

_t_ inevitable provides a reasonable starting point for accident prediction, while _t_ collision marks the last meaningful moment for anticipation. To address the subjective nature of appearance-based alert time annotation, we calculate statistics over annotations from multiple annotators. 

Table 2. Different datasets provide different definition of the temporal annotations. “ ” represents available temproal annotations from the original paper release, while “M” represents reannotation by MMAU [12] (only part of the samples were reannotated). 

||_t_appearing|_t_inevitable|_t_collision|_t_end<br>~~o~~f<br>~~a~~nomaly|
|---|---|---|---|---|
|A3D [56]|M||M|, M|
|DoTA [54, 55]|M||M|, M|
|CTA [57]|-|<sup>4</sup>||-|
|CCD [4]|M|-|, M|M|
|ROL [24]||-||-|
|DADA2000[11]||-|||
|**Nexar**|-|<sup>5</sup>||-|



## **3. Dataset Description** 

Our dataset consists of 1,500 road facing videos recorded by Nexar dashcams. Video recording is triggered by hardbreaks or sudden accelerations detected by the in-camera IMU. Each video has a resolution of 1280×720 at approximately 30 frames per second and an average duration of 40 seconds. 

A team of annotators reviewed hundreds of thousands of videos and labeled them based on the event that triggered the recording as: 

- Collision: when there is an accident event between the ego vehicle and a third party; 

- Near-collision: when there is a close call event between the ego vehicle and a third party; 

- Normal driving: when there is no collision or nearcollision (no event). 

Collisions and near-collisions are considered critical events and are grouped together as positive examples of situations that can lead to an accident, while normal driving videos serve as negative examples. 

Two annotation tasks were conducted: 1) A **general attributes annotation task** , in which all videos were classified based on weather, scene, and lighting conditions. 2) A **incidents specific annotation task** for positive examples, which involved identifying whether the event was visible in 

> 4This is based on the definition of the starting of the “causality”, the moment when the vehicle that caused the eventual accident starts to demonstrate wrongdoing and eccentric behavior. We observe that such definition is slightly late-biased compared to the definition of _t_ inevitable. 

> 5Note our formal definition is the earliest moment a driver should intervene to prevent the accident, which is slightly different from the original definition of _t_ inevitable. 

2585 



<!-- Start of picture text -->
… …<br>t<575 t=575 alert t=578 t=581 accident t>581<br><!-- End of picture text -->

Figure 2. Our dataset contains a considerable fraction of samples where the accident happens within just a couple of frames, a time interval that is far shorter than a human driver could react. The anticipation for such cases is applicable only for autonomous driving scenarios, indicating that our dataset opens up the potential for evaluating algorithms that serve beyond human drivers. 



<!-- Start of picture text -->
CTAD<br>CCD<br>ROL<br>Nexar<br>Real Fog Extreme View Lens Flare Wet Blur<br><!-- End of picture text -->

Figure 3. Our dataset demonstrates significant diversity regarding road types, weather condition, lighting conditions, types of vehicles as well as blurring, lens flare or other artifacts caused by the camera capture process, serving as a more challenging testbed than representative existing datasets for early traffic anticipation (CTAD [38], CCD [4], ROL [24]). 











Figure 4. Our dataset demonstrates diversity regarding video recording with respect to the weather, lighting condition as well as road types. One of the video was recorded indoor and does not have the weather label. 

the video, determining the event time, and annotating the alert time. 

Categorical attributes were annotated by three annotators, with the final value determined as the mode of the annotations. 

Temporal attributes were annotated by 10 different annotators. The final event time was determined as the median value of all annotations, while the final alert time was set as the second-highest value. Given the high subjectivity of alert time annotations, we selected the second-highest value instead of the median to promote shorter alert-to-accident intervals and achieve higher consensus among annotators. The second-highest value also provides greater robustness to outliers compared to the maximum. 

### **3.1. Acceptance criteria** 

The dataset was filtered to include only videos that met the following criteria: 

1. At least 2 annotators agreed on all categorical attributes (e.g. event visibility, weather, scene, lighting conditions); 

2. The video was not corrupted; 

3. For positive cases, we required that the event be visible in the video, involve cars or trucks (excluding pedestrians, bicycles, motorcycles, animals, or stationary objects), and that the (potential) impact occur on the front 

2586 

side of the ego vehicle. 

### **3.2. Video anonymization and data privacy** 

While preparing this dataset, we prioritized the protection of privacy and compliance with ethical standards. To ensure the privacy of drivers and other individuals captured in the videos, we applied the following anonymization measures: 

1. Blurring of faces, license plates, and the dashboard of the ego vehicle; 

2. Removal of audio from all videos; 

3. Exclusion of videos that begin or end within 100 meters or 2 minutes of the start or end of a Nexar user’s ride to prevent potential disclosure of sensitive locations such as home or workplace. 

These measures ensure that the dataset aligns with ethical guidelines for research while minimizing the risk of reidentification or misuse. By openly addressing these considerations, we aim to encourage responsible use of this dataset within the research community. 

### **3.3. Dataset sampling and split** 

At the end of the filtering process, we randomly sampled 750 negative examples and 750 positive examples (400 collisions and 350 near-collisions), resulting in a balanced dataset with 1,500 videos. In addition, we reserved a part of the videos to make a test set (more details in section 4.1). 

### **3.4. Dataset statistics** 

We illustrate the statistical distribution of the dataset in Figure 5. The distribution of video durations is bimodal, with most videos lasting approximately 40 seconds. The duration depends on the dashcam model, as some models record shorter clips. Events (collisions and near-collisions) typically occur near the middle of the video. The average alertto-accident interval is 1.6 seconds, with a maximum of 4.5 seconds. 

Regarding scene, weather, and lighting conditions, our dataset reflects the distribution captured by our cameras, with urban scenes, clear skies, and normal lighting conditions being the most frequent. 

The dataset is available on huggingface<sup>6</sup> and provided subject to the Nexar license<sup>7</sup> 

## **4. Challenge Description** 

The challenge aims to advance the development of accident anticipation models using real-world dashcam footage. Participants should build machine learning models to estimate the likelihood of an imminent collision based on short video segments. Hosted on Kaggle [42], the competition consists 

> 6https://huggingface.co/datasets/nexar-ai/nexar collision prediction 

> 7https://huggingface.co/datasets/nexar- 

> ai/nexar collision prediction/blob/main/LICENSE 

of a single task: predicting whether an accident is about to occur within a given time frame. 

### **4.1. Test set** 

The competition test set is composed of 1344 videos with a duration of approximately 10 seconds. These videos were generated from a pool of 568 videos that follow a distribution similar to that of the training data. For each positive video, up to 3 videos are generated by cropping the original video so that videos have approximately 10 seconds and terminate at instants _t_ event _−_ ΔTTE, where _t_ event is the time of the (near-)collision event in milliseconds and ΔTTE represents different times to event, namely 500ms, 1000ms and 1500ms. This allows testing how models behave at different times to accident. A video is only cropped at a given ΔTTE _>_ 500 when the difference between the annotated event and the alert time is greater than the correspondent ΔTTE. For negative examples, fake event times were generated by adding Gaussian noise to half of the duration of the video, and by generating fake alert times to match the distribution of the positive cases. 

The test set is subdivided into two balanced sets of equal size: public and private. The public test set is used to evaluate model submissions during the competition. The private test set will be used when the competition closes to make the final leader board. Teams continuously submit solutions to the entire test set without knowing the internal division between public and private sets. 

### **4.2. Evaluation metric** 

The goal of the competition is to find solutions that maximize recall, precision, and anticipation time (i.e. detect positive cases as early as possible). We use the mean of the Average Precision (AP) measured at different time-toevent (TTE) values. The AP summarizes the relation between Precision and Recall at a fixed TTE. By combining the AP measured at different TTE values we can summarize the models’ ability to anticipate a potential collision event. 

The output of the model should be a confidence score for each of the videos in the test set. Scores are grouped into three sets according to the time-to-event (TTE) of the video (500 ms, 1000 ms, or 1500 ms). Then, for each group, the Average Precision (AP) is computed. AP measures the area under the precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight (scikitlearn implementation [44]): 



where _Pn_ and _Rn_ are the precision and recall at the nth threshold. Having the AP calculated for each TTE, the mean Average Precision is calculated as the mean of the 

2587 



Figure 5. Histogram of video duration (left), time of event (center), and alert-to-accident interval (right). 

three AP scores: 



## **5. Contributions** 

In this paper, we introduced the Nexar Dashcam Collision Prediction Dataset and Challenge, aimed at advancing research in traffic accident anticipation and autonomous driving safety. 

Our key contributions are as follows. 

- We release to the community, under a permissive license, a new challenging and diverse dataset for training and evaluating algorithms for traffic accident anticipation. The dataset is captured from multiple urban and suburban areas in the US and encompasses a wide range of real-world scenarios, including various weather and lighting conditions, scene types, and dashcam capturing effects. Our clips consist of high-definition (720p) video sequences, each spanning approximately 40 seconds at 30 frames per second, providing rich temporal context that enables models to identify reliable accident predictors. 

- We provide consistent and well-structured temporal annotations tailored for the traffic accident anticipation task. Our dataset includes a significant portion of cases where the accident occurs within just a few frames – far shorter than a human driver’s reaction time – making it particularly valuable for evaluating algorithms in the context of autonomous driving. To address the subjectivity of alert time annotation, we use multiple annotators and extract robust statistics. 

- The challenge establishes a standardized evaluation framework and benchmarking protocol available in Kaggle [42], emphasizing early and reliable predictions through multiple precision-recall curves across different time-to-event intervals. 

## **6. Future Directions** 

We believe this work makes a significant contribution to traffic safety and autonomous vehicle research by fostering innovation in early accident prediction. Future improvements to the dataset could include expanding annotations to capture additional contextual factors such as vehicle types, driver behaviors, and road infrastructure details. Increasing the dataset’s diversity by incorporating footage from different geographic regions and traffic environments would further enhance its robustness. Additionally, integrating edgecase scenarios, such as extreme weather conditions, events involving vulnerable road users, and complex multi-vehicle interactions, would make the dataset more comprehensive for real-world applications. 

Beyond dataset expansion, this dataset can serve as a benchmark for future automated accident prediction systems, particularly for autonomous vehicles and ADAS technologies. The dataset includes a significant number of cases where the alert-to-accident interval is extremely short, making it especially valuable for testing accident prevention in real-time decision-making scenarios. Given that human reaction time is typically at least 0.7 seconds [16], many of these accidents would be unavoidable for human drivers. However, an autonomous vehicle equipped with real-time accident prediction capabilities could anticipate and intervene earlier. Moreover, in a V2X-enabled environment, the detecting vehicle could alert surrounding vehicles, enabling a coordinated response to mitigate or even prevent collisions. This positions our dataset as a foundation for advancing accident prediction and prevention, accelerating progress toward safer, more proactive transportation systems. 

2588 

## **References** 

- [1] Mansoor G Al-Thani, Ziyu Sheng, Yuting Cao, and Yin Yang. Traffic transformer: Transformer-based framework for temporal traffic accident prediction. _AIMS Mathematics_ , 9(5):12610–12629, 2024. 2 

- [2] Mohammad Sadegh Aliakbarian, Fatemeh Sadat Saleh, Mathieu Salzmann, Basura Fernando, Lars Petersson, and Lars Andersson. Viena: A driving anticipation dataset. In _Asian Conference on Computer Vision_ , pages 449–466. Springer, 2018. 1, 3 

- [3] Sonia Baee, Erfan Pakdamanian, Inki Kim, Lu Feng, Vicente Ordonez, and Laura Barnes. Medirl: Predicting the visual attention of drivers via maximum entropy deep inverse reinforcement learning. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 13178– 13188, 2021. 

- [4] Wentao Bao, Qi Yu, and Yu Kong. Uncertainty-based traffic accident anticipation with spatio-temporal relational learning. In _ACM Multimedia Conference_ , May 2020. 1, 2, 3, 4 

- [5] Wentao Bao, Qi Yu, and Yu Kong. Drive: Deep reinforced accident anticipation with visual explanation. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 7619–7628, 2021. 2 

- [6] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In _European Conference on Computer Vision (ECCV)_ , 2020. 2 

- [7] Fu-Hsiang Chan, Yu-Ting Chen, Yu Xiang, and Min Sun. Anticipating accidents in dashcam videos. In _Computer Vision–ACCV 2016: 13th Asian Conference on Computer Vision, Taipei, Taiwan, November 20-24, 2016, Revised Selected Papers, Part IV 13_ , pages 136–153. Springer, 2017. 1, 2, 3 

- [8] Gary-Patrick Corcoran and James Clark. Traffic risk assessment: A two-stream approach using dynamic-attention. In _2019 16th Conference on Computer and Robot Vision (CRV)_ , pages 166–173. IEEE, 2019. 2 

- [9] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Scharw¨achter, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset. In _CVPR Workshop on the Future of Datasets in Vision_ , volume 2, page 1, 2015. 1 

- [10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_ , 2020. 2 

- [11] Jianwu Fang, Lei-Lei Li, Kuan Yang, Zhedong Zheng, Jianru Xue, and Tat-Seng Chua. Cognitive accident prediction in driving scenes: A multimodality benchmark. _arXiv preprint arXiv:2212.09381_ , 2022. 1, 2, 3 

- [12] Jianwu Fang, Lei-lei Li, Junfei Zhou, Junbin Xiao, Hongkai Yu, Chen Lv, Jianru Xue, and Tat-Seng Chua. Abductive ego-view accident video understanding for safe driving perception. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 22030– 

22040, 2024. 1, 2, 3 

- [13] Jianwu Fang, Dingxin Yan, Jiahuan Qiao, Jianru Xue, and Hongkai Yu. Dada: Driver attention prediction in driving accident scenarios. _IEEE transactions on intelligent transportation systems_ , 23(6):4959–4971, 2021. 1, 2, 3 

- [14] Mishal Fatima, Muhammad Umar Karim Khan, and ChongMin Kyung. Global feature aggregation for accident anticipation. In _2020 25th International Conference on Pattern Recognition (ICPR)_ , pages 2809–2816. IEEE, 2021. 2 

- [15] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In _Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2012. 1 

- [16] Marc Green. ” how long does it take to stop?” methodological analysis of driver perception-brake times. _Transportation human factors_ , 2(3):195–216, 2000. 6 

- [17] Kazuki HARADA, Yuta MARUYAMA, Tomonori TASHIRO, and Gosuke OHASHI. Traffic accident prediction without object detection for single-vehicle accidents. _IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences_ , page 2024IMP0005, 2025. 2 

- [18] Roei Herzig, Elad Levi, Huijuan Xu, Hang Gao, Eli Brosh, Xiaolong Wang, Amir Globerson, and Trevor Darrell. Spatio-temporal action graph networks. In _Proceedings of the IEEE/CVF international conference on computer vision workshops_ , pages 0–0, 2019. 3 

- [19] Weiming Hu, Xuejuan Xiao, Dan Xie, and Tieniu Tan. Traffic accident prediction using vehicle tracking and trajectory analysis. In _Proceedings of the 2003 IEEE International Conference on Intelligent Transportation Systems_ , volume 1, pages 220–225. IEEE, 2003. 2 

- [20] Daehyeon Jeong, Hoosang Lee, Juseong Park, and Jeha Ryu. A driver warning system based on anticipation of egoinvolved accident objects. 2 

- [21] Shunsuke Kamijo, Yasuyuki Matsushita, Katsushi Ikeuchi, and Masao Sakauchi. Traffic monitoring and accident detection at intersections. _IEEE transactions on Intelligent transportation systems_ , 1(2):108–118, 2000. 1, 2 

- [22] Muhammad Monjurul Karim, Yu Li, and Ruwen Qin. Toward explainable artificial intelligence for early anticipation of traffic accidents. _Transportation research record_ , 2676(6):743–755, 2022. 2 

- [23] Muhammad Monjurul Karim, Yu Li, Ruwen Qin, and Zhaozheng Yin. A dynamic spatial-temporal attention network for early anticipation of traffic accidents. _IEEE Transactions on Intelligent Transportation Systems_ , 23(7):9590– 9600, 2022. 2 

- [24] Muhammad Monjurul Karim, Zhaozheng Yin, and Ruwen Qin. An attention-guided multistream feature fusion network for early localization of risky traffic agents in driving videos. _IEEE Transactions on Intelligent Vehicles_ , 9(1):1792–1803, 2023. 1, 2, 3, 4 

- [25] Hirokatsu Kataoka, Teppei Suzuki, Shoko Oikawa, Yasuhiro Matsui, and Yutaka Satoh. Drive video analysis for the detection of traffic near-miss incidents. In _2018 IEEE International Conference on robotics and automation (ICRA)_ , pages 3421–3428. IEEE, 2018. 1 

- [26] Hoon Kim, Kangwook Lee, Gyeongjo Hwang, and Changho 

2589 

Suh. Crash to not crash: Learn to identify dangerous vehicles using a simulator. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 33(01):978–985, 2019. 1, 3 

- [27] Hoon Kim, Kangwook Lee, Gyeongjo Hwang, and Changho Suh. Predicting vehicle collisions using data collected from video games. _Machine Vision and Applications_ , 32(4):93, 2021. 1, 2 

- [28] Lei-Lei Li and Jianwu Fang. Traffic accident anticipation via driver attention auxiliary. In _International Conference on Autonomous Unmanned Systems_ , pages 348–360. Springer, 2023. 

- [29] Lei-Lei Li, Jianwu Fang, and Jianru Xue. Cognitive traffic accident anticipation. _IEEE Intelligent Transportation Systems Magazine_ , 2024. 

- [30] Haicheng Liao, Yongkang Li, Zhenning Li, Zilin Bian, Jaeyoung Lee, Zhiyong Cui, Guohui Zhang, and Chengzhong Xu. Real-time accident anticipation for autonomous driving through monocular depth-enhanced 3d modeling. _Accident Analysis & Prevention_ , 207:107760, 2024. 

- [31] Haicheng Liao, Yongkang Li, Chengyue Wang, Yanchen Guan, Kahou Tam, Chunlin Tian, Li Li, Chengzhong Xu, and Zhenning Li. When, where, and what? a benchmark for accident anticipation and localization with large language models. In _Proceedings of the 32nd ACM International Conference on Multimedia_ , pages 8–17, 2024. 

- [32] Haicheng Liao, Haoyu Sun, Huanming Shen, Chengyue Wang, Chunlin Tian, KaHou Tam, Li Li, Chengzhong Xu, and Zhenning Li. Crash: Crash recognition and anticipation system harnessing with context-aware and temporal focus attentions. In _Proceedings of the 32nd ACM International Conference on Multimedia_ , pages 11041–11050, 2024. 2 

- [33] Chunsheng Liu, Zijian Li, Faliang Chang, Shuang Li, and Jincan Xie. Temporal shift and spatial attention-based twostream network for traffic risk assessment. _IEEE Transactions on Intelligent Transportation Systems_ , 23(8):12518– 12530, 2021. 1, 3 

- [34] Wei Liu, Yafei Li, Tao Zhang, Yixiang Gao, Longsheng Wei, and Jun Chen. Ccaf-net: Cascade complementarity-aware fusion network for traffic accident prediction in dashcam videos. _Neurocomputing_ , 624:129285, 2025. 2 

- [35] Wei Liu, Tao Zhang, Yisheng Lu, Jun Chen, and Longsheng Wei. That-net: Two-layer hidden state aggregation based two-stream network for traffic accident prediction. _Information Sciences_ , 634:744–760, 2023. 

- [36] Xinxin Liu, Yuchen Zhou, and Chao Gou. Learning from interaction-enhanced scene graph for pedestrian collision risk assessment. _IEEE Transactions on Intelligent Vehicles_ , 2023. 2 

- [37] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 10012–10022, 2021. 2 

- [38] Haohan Luo and Feng Wang. A simulation-based framework for urban traffic accident detection. In _ICASSP 20232023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 1–5. IEEE, 2023. 1, 3, 4 

- [39] Farhan Mahmood, Daehyeon Jeong, and Jeha Ryu. A new 

approach to traffic accident anticipation with geometric features for better generalizability. _IEEE Access_ , 11:29263– 29274, 2023. 2 

- [40] Arnav Vaibhav Malawade, Shih-Yuan Yu, Brandon Hsu, Deepan Muthirayan, Pramod P Khargonekar, and Mohammad Abdullah Al Faruque. Spatiotemporal scene-graph embedding for autonomous vehicle collision prediction. _IEEE Internet of Things Journal_ , 9(12):9379–9388, 2022. 2 

- [41] Yuta Maruyama and Gosuke Ohashi. Accident prediction model using divergence between visual attention and focus of expansion in vehicle-mounted camera images. _IEEE Access_ , 2023. 2 

- [42] Daniel C. Moura, Shizhan Zhu, and Orly Zvitia. Nexar dashcam crash prediction challenge. https://kaggle.com/ competitions/nexar-collision-prediction, 2025. Kaggle. 5, 6 

- [43] Patrik Patera, Yie-Tarng Chen, and Wen-Hsien Fang. Spatiotemporal adaptation with dilated neighbourhood attention for accident anticipation. In _2024 IEEE International Conference on Image Processing (ICIP)_ , pages 2452–2458. IEEE, 2024. 2 

- [44] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. _Journal of Machine Learning Research_ , 12:2825–2830, 2011. 5 

- [45] Yu Qiao, Xiaohui Yang, and Anran Lu. Msan: Multi-scale adaptive network guided by human attention for accident prediction. In _Pacific Rim International Conference on Artificial Intelligence_ , pages 243–255. Springer, 2024. 2 

- [46] Wenfeng Song, Shuai Li, Tao Chang, Ke Xie, Aimin Hao, and Hong Qin. Dynamic attention augmented graph network for video accident anticipation. _Pattern Recognition_ , 147:110071, 2024. 2 

- [47] Tomoyuki Suzuki, Hirokatsu Kataoka, Yoshimitsu Aoki, and Yutaka Satoh. Anticipating traffic accidents with adaptive loss and large-scale incident db. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3521–3529, 2018. 1, 2 

- [48] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herv´e J´egou. Training data-efficient image transformers & distillation through attention. _International Conference on Machine Learning (ICML)_ , 2021. 2 

- [49] A Vaswani. Attention is all you need. _Advances in Neural Information Processing Systems_ , 2017. 2 

- [50] Ayush Verma and Manju Khari. Vision-based accident anticipation and detection using deep learning. _IEEE Instrumentation & Measurement Magazine_ , 27(3):22–29, 2024. 2 

- [51] Tianhang Wang, Kai Chen, Guang Chen, Bin Li, Zhijun Li, Zhengfa Liu, and Changjun Jiang. Gsc: A graph and spatiotemporal continuity based framework for accident anticipation. _IEEE Transactions on Intelligent Vehicles_ , 9(1):2249– 2261, 2023. 2 

- [52] Tianqi Wang, Sukmin Kim, Ji Wenxuan, Enze Xie, Chongjian Ge, Junsong Chen, Zhenguo Li, and Ping Luo. Deepaccident: A motion and accident prediction benchmark for v2x autonomous driving. _Proceedings of the AAAI Con-_ 

2590 

_ference on Artificial Intelligence_ , 38(6):5599–5606, 2024. 1, 3 

- [53] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, et al. Internvideo2. 5: Empowering video mllms with long and rich context modeling. _arXiv preprint arXiv:2501.12386_ , 2025. 2 

- [54] Yu Yao, Xizi Wang, Mingze Xu, Zelin Pu, Ella Atkins, and David Crandall. When, where, and what? a new dataset for anomaly detection in driving videos. _arXiv preprint arXiv:2004.03044_ , 2020. 1, 2, 3 

- [55] Yu Yao, Xizi Wang, Mingze Xu, Zelin Pu, Yuchen Wang, Ella Atkins, and David J Crandall. Dota: Unsupervised detection of traffic anomaly in driving videos. _IEEE transactions on pattern analysis and machine intelligence_ , 45(1):444–459, 2022. 1, 3 

- [56] Yu Yao, Mingze Xu, Yuchen Wang, David J Crandall, and Ella M Atkins. Unsupervised traffic accident detection in first-person videos. In _2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ , pages 273– 280. IEEE, 2019. 1, 2, 3 

- [57] Tackgeun You and Bohyung Han. Traffic accident benchmark for causality recognition. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VII 16_ , pages 540–556. Springer, 2020. 1, 3 

- [58] Kuo-Hao Zeng, Shih-Han Chou, Fu-Hsiang Chan, Juan Carlos Niebles, and Min Sun. Agent-centric risk assessment: Accident anticipation and risky region localization. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 2222–2230, 2017. 1, 2, 3 

2591 

