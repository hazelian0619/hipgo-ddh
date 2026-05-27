# HipGo: An Intelligent Multi-Module Medical Agent Platform for Remote Diagnosis and Management of Rare Hip Diseases

## Abstract

Developmental Dysplasia of the Hip (DDH) is a rare orthopedic condition with a prevalence of 1-3%, characterized by significant concealment. While traditionally considered a pediatric disease, recent studies reveal a substantial population of undiagnosed adult patients suffering progressive joint degeneration due to missed early diagnosis. We present HipGo, an intelligent multi-module medical agent platform for remote DDH diagnosis and management, specifically designed for adults and adolescents. Our platform innovatively combines CNN-GAT fusion models for precise hip joint keypoint localization, integrates structured diagnostic parameters with deep visual features for multimodal clinical report generation, and incorporates intelligent decision support and remote rehabilitation management modules. We evaluated HipGo on a dataset of 800 adult hip X-ray images collected from social media platforms, demonstrating significant improvements in keypoint detection accuracy and clinical diagnostic consistency. The system provides stable and reliable remote diagnostic services even in resource-limited areas, offering a comprehensive solution from screening to rehabilitation for adult DDH patients.

## 1. Introduction

Developmental Dysplasia of the Hip (DDH) is a rare orthopedic condition with concealed characteristics, affecting 1-3% of the global population [21][22]. DDH not only impacts pediatric populations but recent research indicates that numerous undiagnosed adult patients experience severe functional impairment due to progressive hip joint degeneration, particularly in developing regions with insufficient healthcare resources where delayed diagnosis and treatment remain prominent issues.

Traditional hip joint X-ray keypoint measurement heavily relies on radiologist experience and manual operations, which are both time-consuming and inconsistent. Although artificial intelligence technologies have achieved significant advances in medical image analysis, most existing systems still face challenges in clinical interpretation, making it difficult to effectively support non-orthopedic specialists and patients in understanding complex anatomical parameters such as center-edge angle and Sharp angle.

Furthermore, patient disease management and rehabilitation processes remain fragmented, lacking integrated remote diagnostic platforms. The insufficient closed-loop coordination from diagnosis to treatment and recovery affects patient compliance and treatment outcomes.

Addressing these clinical needs and technical bottlenecks, we propose HipGo—an intelligent multi-module medical agent platform for remote diagnosis and management of adult and adolescent DDH. This platform innovatively combines CNN and Graph Attention Network (GAT) fusion models for precise hip joint keypoint localization, integrates structured diagnostic parameters with deep visual features for multimodal automatic clinical report generation, and incorporates intelligent decision support and remote rehabilitation management modules, aiming to achieve comprehensive intelligent closed-loop management from precise diagnosis to personalized rehabilitation.

Our key contributions include:

1. Utilizing 800 heterogeneous X-ray image datasets collected from real social media platforms, employing active learning strategies to improve data annotation efficiency and model stability;

2. Proposing an anatomically-constrained CNN-GAT multi-dimensional feature fusion model that significantly improves keypoint detection accuracy and clinical angle measurement precision;

3. Designing a multimodal automatic report generation system that enhances diagnostic report clinical interpretability and practicality;

4. Constructing an end-to-end intelligent diagnostic agent platform that achieves seamless online-offline integration, promoting equitable healthcare access.

## 2. Related Work

Recent years have witnessed significant advances in medical artificial intelligence across image analysis, automatic report generation, and intelligent diagnostic platforms. This section reviews research status related to DDH remote intelligent diagnosis from four technical perspectives, providing theoretical foundations for HipGo platform's technical innovations.

### 2.1 Medical Image Keypoint Detection and Spatial Modeling

Medical image keypoint detection serves as the foundation for intelligent diagnosis of orthopedic diseases. Early methods primarily relied on traditional computer vision techniques such as SIFT and HOG features [1], but achieved limited accuracy when processing complex anatomical structures in medical images. With deep learning development, Convolutional Neural Networks (CNNs) became mainstream approaches for keypoint detection.

Multi-scale feature extraction methods based on ResNet and HRNet have demonstrated excellence in medical image analysis [2][4]. HRNet particularly achieved high accuracy in anatomical landmark localization tasks by maintaining high-resolution feature representations. However, pure CNN methods have limitations in modeling spatial relationships among keypoints, struggling to capture global topological information of complex structures like the pelvis.

To address this problem, Graph Neural Networks (GNNs) have been introduced to medical image analysis [5]. Self-supervised keypoint detection methods enhanced feature representation capabilities through deep knowledge distillation [3], while Graph Attention Network (GAT) based methods can dynamically learn dependencies among anatomical points [5][28]. Recent research demonstrates that CNN-GNN fusion architectures significantly improved spatial consistency of keypoint detection in tasks such as brain MRI registration [7] and fundus image analysis [4].

Nevertheless, existing methods primarily target standardized medical images, with insufficient adaptability to low-quality images in non-standard environments such as social media, and lack specialized anatomical constraint modeling for rare diseases like DDH.

### 2.2 Medical Report Generation and Multimodal Fusion

Medical report generation has undergone technical evolution from template-driven to deep learning approaches. Early systems primarily relied on rule templates and expert knowledge bases, ensuring medical terminology accuracy but with limited flexibility and personalization [8].

Recent encoder-decoder architecture-based deep learning methods have become mainstream [8][9]. These methods achieve end-to-end mapping from medical images to text reports through visual feature extraction and sequence generation models. Multimodal fusion techniques further improved report generation quality by combining image features, structured data, and clinical context information to generate more accurate and complete diagnostic reports [6].

Self-training frameworks like REMOTE demonstrated effectiveness in enhancing report generation models using unlabeled data, achieving performance comparable to fully supervised methods using only 1% labeled data [6]. Large Language Model (LLM) introduction brings new opportunities for medical report generation, particularly in improving report readability and clinical interpretability [8].

However, existing methods primarily focus on common imaging types such as chest X-rays, with limited automatic interpretation capabilities for complex geometric parameters (such as CE angle and Sharp angle) required for rare diseases like DDH, struggling to meet precision requirements for specialist diagnosis.

### 2.3 Intelligent Medical Agents and Multi-Agent Systems

Intelligent medical agent systems represent an important development direction in medical informatization. Multi-Agent Systems (MAS) achieve intelligent management of medical processes by simulating collaboration among doctors, patients, and medical devices [10][11]. These systems possess distributed, autonomous, and heterogeneous characteristics, adapting to healthcare environment complexity and diversity.

In practical applications, multi-agent AI systems demonstrate significant advantages in clinical decision support, personalized treatment planning, and real-time monitoring [11]. Systems improve diagnostic accuracy, treatment planning effectiveness, and cross-departmental coordination efficiency through agent collaboration. However, issues such as data bias, insufficient interoperability, and responsibility attribution constrain large-scale deployment of such systems [11].

Agent-based medical data real-time processing systems provide new approaches for handling complex medical information [12], but existing systems mostly focus on general medical scenarios, lacking specialized design for rare disease diagnosis and treatment special needs and comprehensive closed-loop management capabilities.

### 2.4 Rare Disease Remote Management Platforms

Rare disease remote management has gained attention in digital health due to patient dispersion and specialist resource scarcity. The European Commission developed Clinical Patient Management System 2.0 (CPMS 2.0) providing cross-border remote consultation platforms for 24 European Reference Networks (ERNs), supporting diagnostic collaboration for over 30 million rare disease patients [13]. This platform achieved European-wide expert resource sharing through secure online consultations, significantly improving rare disease patient diagnostic accessibility.

Remote Patient Monitoring (RPM) technology plays important roles in rare disease management [14]. Through wearable devices, mobile health applications, and cloud platforms, RPM achieves continuous collection and real-time analysis of patient health data. This technology particularly suits rare disease patients requiring long-term monitoring, providing early warning, personalized intervention, and treatment compliance monitoring [14].

Digital treatment platforms like Mahalo Health provide precision management solutions specifically for rare disease complex symptoms [15], while OpenApp's patient registry platform Clinical Insight has been deployed across multiple rare disease patient organizations, from ultra-rare diseases with fewer than 100 patients to larger patient populations [16].

Despite progress in rare disease remote management, most existing platforms still exhibit limitations such as single functionality, lack of intelligent diagnostic capabilities, and limited patient interaction experiences, struggling to provide comprehensive intelligent services from screening diagnosis to rehabilitation management.

### 2.5 Technical Development Trends and Innovation Gaps

Comprehensive analysis of existing research reveals medical AI technology is developing toward multimodal fusion, intelligent collaboration, and personalized services. However, significant technical gaps remain for remote diagnosis of orthopedic rare diseases like DDH: (1) lack of robust keypoint detection methods adapting to non-standard imaging environments such as social media; (2) insufficient clinical interpretation capabilities of existing report generation systems for complex geometric parameters; (3) absence of integrated end-to-end intelligent platforms incorporating image analysis, diagnostic decision-making, and rehabilitation management.

HipGo platform aims to fill these technical gaps through CNN-GAT fusion anatomical constraint modeling, multimodal clinical report generation, and comprehensive intelligent agent collaboration, providing systematic solutions for remote intelligent diagnosis of rare diseases like DDH.

## 3. System Architecture and End-to-End Diagnostic Workflow Design

This system addresses the comprehensive requirements of DDH diagnostic workflows by constructing a closed-loop end-to-end intelligent architecture centered on "perception-analysis-decision-execution-feedback" (see Figure 1). The system encompasses data collection, image analysis, report generation, decision support, rehabilitation management, and information integration, spanning online and offline diagnostic scenarios to achieve efficient collaboration among patients, physicians, and intelligent systems while ensuring closed-loop synchronization of diagnostic information.

**[FIGURE 1 PLACEHOLDER: System Architecture Diagram showing the end-to-end workflow from patient image upload to report generation]**

### 3.1 End-to-End Diagnostic Workflow

Patients first input basic information, annotate pain locations and severity, and upload X-ray images through mobile interfaces. The system automatically performs standardized data preprocessing to ensure accuracy and consistency in subsequent analysis. The image analysis module, based on CNN-GAT fusion models, achieves automatic detection of hip joint keypoints and spatial relationship modeling, outputting high-confidence structural parameters. The report generation module integrates structured parameters with deep image features, combined with Large Language Models (LLMs), to automatically generate professional and interpretable clinical reports. The decision support module provides personalized treatment recommendations based on diagnostic results and clinical guidelines. The rehabilitation management module dynamically formulates and adjusts staged rehabilitation plans based on diagnosis and patient feedback, continuously tracking rehabilitation progress. All diagnostic data and reports synchronize to Electronic Health Records (EHR), supporting multi-platform sharing and export.

The system supports patients in receiving intelligent workflow guidance, personalized information delivery, and online feedback throughout diagnostic stages. Physicians can seamlessly switch between remote healthcare and hospital services, achieving collaboration between offline diagnosis and intelligent agent remote assistance, improving diagnostic efficiency and accessibility.

### 3.2 Core Module Design

The **Data Collection and Preprocessing** module provides intuitive interactive interfaces supporting patient self-annotation of pain information, automatic image quality detection and enhancement, ensuring data completeness and standardization.

The **Image Analysis** module employs CNN-GAT fusion models combined with multi-dimensional spatial features to achieve high-precision keypoint detection and abnormal region annotation, providing quantitative evidence for diagnosis.

The **Report Generation** module performs multimodal feature fusion based on Transformer and LLM architectures to automatically generate structured, traceable diagnostic reports, incorporating expert feedback and continuous optimization.

The **Decision Support** module outputs personalized treatment recommendations based on clinical knowledge bases and individual patient information, supporting shared decision-making between physicians and patients while ensuring scientific rigor and safety.

The **Rehabilitation Management** module automatically formulates and dynamically adjusts rehabilitation plans based on diagnosis and feedback, providing staged guidance for patient rehabilitation training to improve rehabilitation efficiency.

The **Information Integration and Closed-Loop Optimization** module achieves data flow through standardized interfaces across modules, with all outputs including confidence levels, abnormality markers, and follow-up recommendations. The system continuously collects patient feedback and clinical data, dynamically optimizing model parameters and decision strategies to form a self-learning, scalable diagnostic system.

### 3.3 Collaborative Mechanisms and Data Security

The system supports seamless switching between online intelligent follow-up and offline expert diagnosis, combining remote healthcare with hospital services to significantly improve diagnostic accessibility and efficiency. All data employs AES-256 encryption for storage and transmission, strictly complying with HIPAA and other medical data security regulations. The architecture possesses good scalability, flexibly adapting to other orthopedic diseases or multi-center remote diagnostic scenarios, providing practical foundations for continuous evolution of intelligent medical systems.

## 4. Methods

We present a CNN-GAT fusion-based intelligent DDH diagnostic system that establishes an end-to-end diagnostic pipeline from "image analysis → report generation → decision support," overcoming limitations in complex anatomical structure modeling and clinical interpretability of traditional approaches.

**[FIGURE 2 PLACEHOLDER: Technical architecture diagram showing CNN-GAT fusion model components]**

### 4.1 Data Collection and Active Learning

To authentically reflect real-world remote healthcare scenarios in China's grassroots regions, this study selected the mainstream domestic social platform "Xiaohongshu" as the primary data collection channel. Users, lacking professional medical equipment and conditions, commonly photograph electronic X-ray images with mobile phones and upload them to comment sections seeking remote diagnostic advice. This data collection approach aligns with patient image acquisition habits in non-professional environments while simulating real-world remote healthcare scenarios with unstable image quality and noisy information transmission.

Based on this data source, we collected 800 user-uploaded adult pelvic anteroposterior X-ray images (gender-independent), annotated case-by-case by two experienced orthopedic specialists. Annotation content encompasses nine critical anatomical points: anterior and posterior acetabular rims, femoral head center, greater and lesser trochanters, iliac crest apex, anterior superior sacral margin, superior pubic symphysis margin, and inferior ischial margin. This keypoint set is based on anatomical landmarks required for three core clinical angle measurements in DDH diagnosis—Center-Edge angle (CE angle), Sharp angle, and Tönnis angle—ensuring clinical relevance and theoretical support of annotations [21][22].

Considering high cost and time consumption of medical image annotation, this study introduced active learning strategies [23], prioritizing samples with lowest confidence for expert review and supplementary annotation through model prediction uncertainty sampling, maximizing utilization of limited annotation resources. The specific workflow involved training a basic model with initial 200 expert-annotated cases, then performing confidence assessment on subsequently collected 600+ images, screening samples with highest uncertainty for additional annotation until model performance stabilized. Finally, the dataset expanded to 800 cases, significantly improving keypoint detection model robustness in real remote healthcare scenarios.

To ensure training data effectiveness and quality, the system established a model prediction confidence threshold of 0.7. This threshold, based on empirical analysis of model prediction confidence distributions, balances sample representativeness and annotation quality, effectively filtering low-confidence noise samples while reducing negative impacts on model training and maintaining sufficient sample diversity to ensure model generalization capability.

### 4.2 Data Preprocessing and Enhancement

Considering privacy sensitivity of social media data, the system employs OCR-based automatic text detection algorithms [24] to identify and mask patient identity information, ensuring data anonymization. All images are uniformly adjusted to 512×512 resolution, keypoint coordinates normalized to intervals, and pixel values standardized using ImageNet statistical parameters [26] to fully utilize pre-trained network feature representation capabilities.

To reduce domain gaps between training data and real application scenarios, this study designed targeted augmentation strategies based on medical image enhancement best practices [25]. Geometric transformations simulate posture variations during mobile photography through random rotation (±15°), translation, and scaling, while photometric transformations introduce Gaussian noise, motion blur, and color jitter to effectively simulate image degradation under different devices and environmental conditions. Validation sets undergo only basic standardization processing, strictly following standard evaluation protocols for medical AI research.

### 4.3 CNN-GAT Fusion Keypoint Detection and Spatial Modeling

Medical image keypoint detection technology is undergoing deep transformation from local feature learning toward structured relationship modeling, particularly evident in DDH diagnosis. Traditional CNN methods, while capable of precise single anatomical point localization, struggle to ensure spatial consistency across regional keypoints—precisely the core requirement for calculating clinical indicators like CE angle and Sharp angle. Existing research shows that Graph Convolutional Networks relying purely on geometric distances or multimodal fusion methods with fixed templates easily produce anatomically unreasonable predictions in complex cases. Therefore, this study proposes an anatomically-guided CNN-GAT fusion architecture establishing new paradigms for structured medical image analysis through deep collaboration between visual perception and spatial reasoning.

#### 4.3.1 Theoretical Foundation

Model design strictly follows DDH diagnostic clinical workflows: orthopedic surgeons must identify 9 critical anatomical points on pelvic anteroposterior X-rays and calculate clinical indicators like CE angle and Sharp angle through their spatial relationships. This process requires detection systems with dual capabilities—capturing local textural details while maintaining global structural constraints. Based on this, the architecture employs dual-branch collaborative design:

**CNN Branch**: ResNet-50 serves as the backbone network, with residual structures [2] effectively alleviating gradient vanishing problems and adapting to medical data scarcity. Feature Pyramid Network (FPN) achieves multi-scale feature fusion through P2-P5 levels (resolutions 1/4 to 1/32), outputting 256-dimensional feature vectors [27]. This design inherits multi-scale theory from Lin et al., effectively addressing scale variations in mobile-captured images: high-level features capture overall pelvic morphology while low-level features preserve details like acetabular margins.

**GAT Branch**: Constructs dynamic graph structures containing 9 keypoints, with adjacency matrices defined by combining anatomical connections (such as mandatory connections between anterior acetabular rim and femoral head center) and adaptive distance thresholds (0.3). Edge features encode as 32-dimensional vectors containing normalized Euclidean distances (8 dimensions), relative angular sine-cosine encodings (16 dimensions), and predefined semantic labels (8 dimensions). This design transcends traditional GCN geometric limitations by deeply integrating anatomical semantic information into graph representation learning [29], enabling models to distinguish between "anatomical connections" and "spatial proximity" relationship differences.

#### 4.3.2 Key Technical Implementation

**Edge Feature-Enhanced Graph Attention Mechanism**: Building upon standard GAT [5][28], we introduce edge feature-participating attention computation:

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j \| \mathbf{W}_e\mathbf{e}_{ij}]))}{\sum_{k \in \mathcal{N}_i} \exp(\cdot)}$$

where $\mathbf{W}_e \in \mathbb{R}^{32 \times 128}$ maps edge features to attention space, and $\mathbf{e}_{ij}$ represents 32-dimensional edge feature vectors. This design enables models to dynamically assess importance of different relationship types: when processing acetabular dysplasia with subluxation cases, attention weights for anatomical connections (such as acetabulum-femoral head) significantly exceed purely spatial proximity relationships.

**Dynamic Gating Fusion Strategy**: We design lightweight gating networks balancing 256-dimensional CNN features with 64-dimensional GAT representations:

$$\mathbf{g} = \sigma(\mathbf{W}_g[\text{Proj}_{256 \rightarrow 64}(\mathbf{F}_{cnn}) \| \mathbf{F}_{gat}] + \mathbf{b}_g)$$
$$\mathbf{F}_{fusion} = \mathbf{g} \odot \mathbf{F}'_{cnn} + (1-\mathbf{g}) \odot \mathbf{F}_{gat}$$

Gating network parameters comprise only 0.8% of total model parameters, with design inspiration from parameter efficiency principles in neural architecture search. When input image quality deteriorates, gating networks automatically increase GAT feature weights, utilizing spatial relationship constraints to compensate for local feature uncertainty.

**Anatomically-Constrained Multi-Task Learning**: Loss functions integrate keypoint regression loss with medical constraint terms:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{keypoint} + \lambda_2 \mathcal{L}_{medical}$$

where $\mathcal{L}_{medical}$ constrains predicted keypoints through backpropagation to generate clinically reasonable angle measurements, deeply embedding clinical knowledge into learning processes and continuing medical AI interpretability framework concepts proposed by Raghu et al. [30], avoiding anatomical abnormalities caused by pure data-driven approaches.

#### 4.3.3 Training Optimization and Engineering Practice

Under NVIDIA RTX 3090 hardware environments, we employ 4-step gradient accumulation achieving effective batch size 16, alleviating BatchNorm statistical bias from small batch training. The optimizer selects Adam (initial learning rate 3e-4, weight decay 1e-4), with adaptive characteristics effectively addressing medical data distribution imbalances. Early stopping mechanisms (patience value 7) and Dropout (p=0.1) work together preventing overfitting to noisy data, ensuring model robustness in real scenarios.

### 4.4 Multimodal Fusion and Clinical Report Generation

After completing keypoint detection and spatial relationship modeling, we applied classical anatomical spatial relationship principles to transform model-output keypoint coordinates into clinically-standard geometric indicators, further generating structured reports with clinical diagnostic value and medical standard compliance.

Traditional angle measurement relies on manual point placement, exhibiting subjective errors and efficiency bottlenecks [21][22]. This study strictly follows anatomical principles, precisely converting model-output 9 keypoint coordinates to CE angle, Sharp angle, and Tönnis angle. For example, CE angle calculation involves not only connecting lines between femoral head center and acetabular outer rim but also requires standard perpendicular line construction to accurately reflect acetabular coverage degree.

Clinical report generation requires integrating information from different modalities: structured angular parameters (CE angle, Sharp angle, Tönnis angle values), deep visual features (256-dimensional feature vectors output by CNN-GAT), and patient clinical context (age, gender, symptom descriptions). This multimodal fusion necessity stems from inherent limitations of single-modality information—purely relying on numerical parameters lacks image detail support, while visual features struggle to directly map to clinical terminology.

To overcome this challenge, this study leverages recent advances in multimodal learning [31][32], employing Cross-Attention mechanisms to achieve deep semantic alignment of heterogeneous information, enabling models to autonomously learn complex dependencies among different modality information. For example, specific CE angle values may correspond to specific visual manifestations of insufficient acetabular coverage in images, significantly enhancing comprehensive understanding of complex clinical scenarios.

Report templates were collaboratively designed by senior orthopedic specialist teams, strictly following DDH clinical diagnostic guidelines, encompassing core modules including "measurement results," "imaging abnormality descriptions," "diagnostic assessments," and "preliminary treatment recommendations." During text generation, we selected medical domain pre-trained LLaVA-Med variants [33] as base models, using fused multimodal features $\mathbf{F}_{aligned}$ and structured report templates as conditional inputs for LLMs. Additionally, to enhance report clinical interpretability, the system overlays angle measurement lines, keypoint annotations, and abnormal region highlights on images, helping physicians and patients quickly verify automatic measurements and diagnostic results.

## 5. Experimental Design and Validation

### 5.1 Research Questions

This study addresses four critical research questions:

1. Can the CNN-GAT fusion model achieve superior keypoint detection accuracy on low-quality social media X-ray images compared to existing baseline methods?

2. What are the specific contributions of core model components (multi-scale features, edge features, dynamic fusion) to performance improvements?

3. How does the consistency between multimodal fusion-generated diagnostic reports and orthopedic expert judgments compare?

4. Does the system's computational efficiency meet practical deployment requirements for remote healthcare?

### 5.2 Dataset and Preprocessing

Our dataset strictly simulates real remote healthcare scenarios, sourced entirely from publicly uploaded adult pelvic anteroposterior X-ray images on mainstream social platforms, comprising 800 samples. These images exhibit high heterogeneity, encompassing different imaging devices, non-standard perspectives, and complex noise interference (such as motion blur, low resolution).

Data annotation was completed by two senior orthopedic specialists strictly following DDH diagnostic standards [21][22], covering 9 critical anatomical points including anterior/posterior acetabular rims and femoral head centers. Addressing challenges of high annotation costs and quality fluctuations in social media data, we employed active learning strategies to optimize annotation efficiency: initially annotating 200 high-quality images to train the base model, then screening samples with maximum information content through model uncertainty sampling for expert review (confidence threshold 0.7), ultimately expanding to 800 annotated cases, improving model generalization in complex cases.

The dataset was divided according to patient independence principles into training set (640 cases), validation set (80 cases), and test set (80 cases). During preprocessing, all images were uniformly adjusted to 512×512 resolution, automatically masked for patient privacy information through OCR technology, then standardized using ImageNet statistical parameters [26]. The training set introduced augmentation strategies including random rotation (±15°), Gaussian noise (σ=0.1), and motion blur (kernel size 15×15), while validation and test sets underwent only basic standardization, strictly following medical AI evaluation protocols.

### 5.3 Experimental Results and Analysis

We comprehensively evaluated HipGo system performance on the 800-case social media X-ray dataset. The dataset was divided according to patient independence principles into training set (640 cases), validation set (80 cases), and test set (80 cases), strictly simulating real remote healthcare scenario complexity and diversity.

Keypoint detection performance evaluation results demonstrate that our CNN-GAT fusion model significantly outperforms baseline methods across all evaluation metrics. PCK@0.1 achieved 0.86, representing 14 percentage point improvement over ResNet-50, 11 percentage point improvement over traditional GAT, and 8 percentage point improvement over HRNet. PCK@0.2 and PCK@0.5 metrics achieved 0.93 and 0.98 respectively, with mAP reaching 0.91, indicating that anatomically-constrained graph attention mechanisms effectively improved spatial consistency of keypoint detection. Particularly when processing low-quality social media images, our method demonstrated superior robustness and generalization capabilities.

Angular measurement accuracy validation confirmed model reliability in clinical indicator calculations. For three critical clinical indicators (CE angle, Sharp angle, and Tönnis angle), our method achieved mean absolute errors of 0.8°, 1.1°, and 0.6° respectively, highly consistent with expert measurements, with correlation coefficients reaching 0.94, 0.91, and 0.96. Clinical accuracy rates were 92%, 89%, and 94% respectively, fully meeting DDH diagnostic clinical standard requirements. These results confirm that CNN-GAT fusion models accurately capture spatial relationships among anatomical structures, providing reliable quantitative evidence for DDH diagnosis.

Computational efficiency analysis demonstrates system practical deployment capabilities. Under NVIDIA RTX 3090 environments, average processing time per image was 2.3 seconds, meeting remote healthcare real-time application requirements. Model parameters totaled 28.7M, representing 10.6% reduction compared to HRNet while maintaining higher detection accuracy. During 7-day continuous operation testing, system availability reached 96.5% with average response time of 2.3 seconds, fully satisfying clinical application stability requirements.

**Table 1: Keypoint Detection Performance Comparison**

| Method | PCK@0.1 | PCK@0.2 | PCK@0.5 | mAP | Parameters |
|--------|----------|----------|----------|-----|------------|
| ResNet-50 [2] | 0.72 | 0.85 | 0.94 | 0.83 | 25.6M |
| HRNet [4] | 0.78 | 0.89 | 0.96 | 0.87 | 32.1M |
| Traditional GAT [5] | 0.75 | 0.86 | 0.95 | 0.85 | 18.3M |
| **CNN-GAT Fusion** | **0.86** | **0.93** | **0.98** | **0.91** | **28.7M** |

**Table 2: Angular Measurement Accuracy Comparison**

| Angle Type | Expert Measurement | AI Measurement | MAE | Correlation | Clinical Accuracy |
|------------|-------------------|----------------|-----|-------------|------------------|
| CE Angle [21] | 25.3°±3.2° | 25.1°±3.1° | 0.8° | 0.94 | 92% |
| Sharp Angle [22] | 42.1°±2.8° | 42.3°±2.9° | 1.1° | 0.91 | 89% |
| Tönnis Angle | 8.7°±1.5° | 8.9°±1.6° | 0.6° | 0.96 | 94% |

### 5.4 Ablation Studies

To deeply understand specific contributions of each component to performance improvements, we conducted systematic ablation experiments. Results demonstrate that each component makes important contributions to final performance, validating design decision effectiveness.

The CNN branch, serving as the foundation for visual feature extraction, achieved PCK@0.1 of 0.72 when used alone, indicating effective capture of local textural details. However, lacking spatial relationship modeling led to insufficient consistency among keypoints, with angular measurement MAE reaching 2.3°, struggling to meet clinical precision requirements.

The GAT branch achieved relatively lower performance when used alone, with PCK@0.1 of 0.68 and angular measurement MAE of 2.8°. This indicates limited pure graph structure modeling capabilities without visual features, struggling to accurately identify anatomical structures.

Edge feature enhancement contributed significantly to performance improvements. Removing edge features resulted in PCK@0.1 declining to 0.81 and angular measurement MAE increasing to 1.5°, confirming the importance of multi-dimensional edge features for capturing anatomical relationships. The 32-dimensional edge feature encoding vector containing normalized Euclidean distances, relative angular encodings, and semantic labels effectively distinguished between "anatomical connections" and "spatial proximity" relationship differences.

Anatomical constraint terms are crucial for ensuring clinical reasonableness. Removing anatomical constraints increased angular measurement MAE by 0.3°, indicating that clinical knowledge embedding plays important roles in avoiding anatomical abnormalities caused by pure data-driven approaches. Anatomical constraints constrain predicted keypoints through backpropagation to generate clinically reasonable angle measurements.

Dynamic fusion strategies improved performance by 2 percentage points compared to fixed weight fusion. Gating network parameters comprise only 0.8% of total model parameters, automatically adjusting CNN and GAT feature weight ratios based on input image quality while maintaining accuracy and achieving parameter efficiency.

**Table 3: Ablation Study Results**

| Component Configuration | PCK@0.1 | PCK@0.2 | Angular MAE | Parameters |
|------------------------|----------|----------|-------------|------------|
| CNN Only [2] | 0.72 | 0.85 | 2.3° | 25.6M |
| GAT Only [5] | 0.68 | 0.82 | 2.8° | 18.3M |
| CNN+GAT (No Edge Features) | 0.81 | 0.90 | 1.5° | 28.7M |
| CNN+GAT (No Constraints) | 0.83 | 0.91 | 1.2° | 28.7M |
| Fixed Fusion Weights | 0.84 | 0.92 | 1.0° | 28.7M |
| **Complete Model** | **0.86** | **0.93** | **0.9°** | **28.7M** |

### 5.5 Clinical Validation

Clinical validation serves as the critical component for evaluating system practical application value. We invited 3 orthopedic specialists with extensive DDH diagnostic experience to independently evaluate system-generated diagnostic reports, ensuring objectivity and reliability of validation results.

Expert consistency evaluation results show that expert-AI diagnostic consistency kappa coefficient reached 0.87, with accuracy of 91%, sensitivity of 88%, and specificity of 94%. These indicators demonstrate that our system can generate diagnostic reports meeting clinical standards with practical clinical application potential. Three expert evaluation results were highly consistent, with kappa coefficients of 0.87, 0.89, and 0.85 respectively, confirming system diagnostic stability and reliability.

Report quality evaluation further validated system clinical practicality. Experts scored AI-generated reports on clinical relevance, readability, and completeness, with average scores of 4.2, 4.0, and 4.1 respectively, indicating report quality met clinical application standards. Particularly in clinical relevance, experts believed the system could accurately identify DDH key characteristics, with generated diagnostic recommendations conforming to clinical guideline requirements.

System usability testing confirmed technical solution feasibility. During 7-day continuous operation testing, system availability reached 96.5% with average response time of 2.3 seconds, fully meeting remote healthcare real-time application requirements. The system performed stably when processing different quality images, demonstrating good robustness against noise and blur.

**Table 4: Expert Consistency Evaluation Results**

| Evaluation Metric | Expert 1 | Expert 2 | Expert 3 | Average |
|------------------|----------|----------|----------|---------|
| Diagnostic Consistency | 0.87 | 0.89 | 0.85 | 0.87 |
| Accuracy | 91% | 93% | 89% | 91% |
| Sensitivity | 88% | 90% | 86% | 88% |
| Specificity | 94% | 96% | 92% | 94% |

## 6. Discussion

### 6.1 Technical Contribution Analysis

This study's primary technical contributions include: (1) First application of social media data to DDH diagnosis, addressing data acquisition challenges in real remote healthcare scenarios; (2) Proposing anatomically-constrained CNN-GAT fusion architecture that effectively improves spatial consistency of keypoint detection; (3) Designing multimodal report generation systems that enhance clinical interpretability of AI diagnosis; (4) Constructing comprehensive intelligent agent systems achieving closed-loop management from diagnosis to rehabilitation.

The CNN-GAT fusion approach represents a significant advancement in medical image analysis. Unlike previous methods that treat keypoint detection as independent localization tasks, our approach explicitly models anatomical relationships through graph attention mechanisms. The edge feature enhancement allows the model to distinguish between meaningful anatomical connections and incidental spatial proximity, leading to more clinically relevant predictions. This is particularly important for DDH diagnosis, where spatial relationships between acetabular and femoral structures are critical for accurate angle measurements.

Our multimodal report generation system addresses a crucial gap in existing medical AI systems. While many systems can detect abnormalities, few can provide clinically meaningful explanations that support decision-making. By integrating structured measurements, visual features, and clinical context through cross-attention mechanisms, our system generates reports that are both technically accurate and clinically interpretable.

### 6.2 Clinical Significance Discussion

Our system demonstrates important clinical significance across multiple dimensions: (1) Significantly improving DDH early diagnosis rates, particularly in medically underserved regions; (2) Reducing diagnostic costs and time while improving diagnostic efficiency; (3) Providing convenient remote diagnostic services for patients, improving healthcare access; (4) Offering reliable AI-assisted tools for physicians, supporting shared decision-making between doctors and patients.

The use of social media data represents a paradigm shift in medical AI development. Traditional medical AI systems rely on curated datasets from established medical institutions, which may not reflect the real-world conditions where these systems will be deployed. By training on images captured with mobile phones and shared on social platforms, our system is inherently adapted to the quality and characteristics of images that patients actually generate in remote healthcare scenarios.

The clinical validation results demonstrate that our system achieves expert-level performance in DDH diagnosis. The high inter-rater agreement (κ = 0.87) between AI and expert diagnoses, combined with strong sensitivity (88%) and specificity (94%), indicates that the system can serve as a reliable diagnostic aid. This is particularly valuable for DDH, where specialized expertise is often unavailable in remote areas.

### 6.3 Limitation Analysis

This study has the following limitations: (1) Dataset primarily sourced from social media may introduce selection bias; (2) System has not yet undergone large-scale validation in real clinical environments; (3) Processing capabilities for complex cases require further improvement; (4) Long-term system stability and security need additional validation.

The reliance on social media data, while innovative, introduces potential biases. Users who share medical images on social platforms may not be representative of the broader patient population. Additionally, the quality and diversity of images available through this channel may be limited compared to comprehensive clinical datasets. Future work should incorporate data from multiple sources to ensure broader generalizability.

The current system focuses specifically on DDH diagnosis and may not generalize to other orthopedic conditions without significant adaptation. While the underlying CNN-GAT architecture is potentially applicable to other anatomical regions, the specific edge features, anatomical constraints, and clinical interpretation modules are tailored to hip joint analysis.

### 6.4 Future Research Directions

Future research directions include: (1) Expanding dataset size and incorporating more clinical scenarios; (2) Optimizing model architecture to improve complex case processing capabilities; (3) Conducting multi-center clinical validation to assess real-world application effectiveness; (4) Exploring generalizability to other orthopedic diseases and expanding application scope.

The integration of longitudinal patient data represents a promising avenue for future development. By tracking patient outcomes over time, the system could learn to predict disease progression and optimize treatment recommendations. This would require establishing long-term partnerships with healthcare institutions to collect follow-up data.

Advanced uncertainty quantification methods could further improve system reliability. While our current approach uses prediction confidence thresholds, more sophisticated approaches could provide clinicians with better understanding of diagnostic uncertainty and guide appropriate follow-up actions.

The development of federated learning approaches could enable system improvement while preserving patient privacy. This would allow the system to learn from data across multiple institutions without requiring centralized data sharing, addressing both privacy concerns and data scarcity issues.

## 7. Conclusion

This study presents HipGo: an intelligent multi-module medical agent platform for remote DDH diagnosis and management. Through CNN-GAT fusion technology, we achieved high-precision hip joint keypoint detection; through multimodal fusion, we generated clinically interpretable diagnostic reports; through intelligent agent systems, we constructed comprehensive management from diagnosis to rehabilitation.

Experimental results on an 800-case social media X-ray dataset demonstrate that our system significantly improves keypoint detection accuracy and clinical diagnostic consistency, providing new technical solutions for intelligent remote diagnosis of rare diseases like DDH. The system achieves expert-level diagnostic performance while maintaining computational efficiency suitable for deployment in resource-limited settings.

The key innovations of our approach include: (1) Novel use of social media data to reflect real-world telemedicine conditions; (2) Anatomically-constrained CNN-GAT fusion that captures both local features and global structural relationships; (3) Multimodal report generation that bridges the gap between AI predictions and clinical interpretation; (4) End-to-end intelligent agent architecture that supports the complete diagnostic workflow.

Our clinical validation demonstrates that the system can generate diagnostic reports with 91% accuracy compared to expert assessments, with high consistency across multiple expert evaluators. The system's ability to process images in 2.3 seconds while maintaining 96.5% availability makes it suitable for real-time clinical deployment.

This work represents a significant step toward democratizing access to specialized orthopedic expertise through AI-powered remote diagnosis. By making DDH diagnosis more accessible and reliable, particularly in underserved regions, our system has the potential to improve early detection rates and patient outcomes for this rare but impactful condition.

Future work will focus on large-scale clinical validation and broader application to other orthopedic conditions, continuing to advance the field of intelligent remote healthcare diagnosis and management.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Extract and translate Introduction section from Chinese text", "status": "completed", "priority": "high"}, {"id": "2", "content": "Translate Related Work section with proper citations", "status": "in_progress", "priority": "high"}, {"id": "3", "content": "Integrate existing Methods section with full manuscript", "status": "pending", "priority": "medium"}, {"id": "4", "content": "Translate Experimental Design and Results sections", "status": "pending", "priority": "high"}, {"id": "5", "content": "Translate Discussion and Conclusion sections", "status": "pending", "priority": "medium"}, {"id": "6", "content": "Format complete reference list with academic citations", "status": "pending", "priority": "medium"}]