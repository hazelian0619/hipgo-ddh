# Methods

We present a CNN-GAT fusion-based intelligent DDH diagnostic system that establishes an end-to-end diagnostic pipeline from "image analysis → report generation → decision support," overcoming limitations in complex anatomical structure modeling and clinical interpretability of traditional approaches.

## 3.1 System Architecture

The system employs modular design comprising four core components: image preprocessing, keypoint detection, multimodal fusion, and report generation. Patients upload pelvic X-ray images through mobile interfaces, enabling automated detection of nine critical anatomical landmarks, clinical angle calculations (CE angle, Sharp angle, Tönnis angle), and structured diagnostic report generation, thereby achieving intelligent and accessible DDH diagnosis.

**[FIGURE 1 PLACEHOLDER: System Architecture Diagram showing the end-to-end workflow from patient image upload to report generation]**

## 3.2 CNN-GAT Fusion for Keypoint Detection

### 3.2.1 Dual-Branch Collaborative Architecture

Addressing spatial dependencies among keypoints in DDH diagnosis, we designed a CNN-GAT dual-branch fusion model. The CNN branch employs ResNet-50 backbone with Feature Pyramid Network (FPN) for multi-scale feature extraction, generating 256-dimensional visual feature vectors that capture local textural details including acetabular margins and femoral head contours. The GAT branch constructs a dynamic 9-node graph structure where nodes correspond to critical anatomical landmarks, with edge weights incorporating anatomical connections and spatial distances. We introduce 32-dimensional edge feature encoding encompassing normalized Euclidean distances, relative angular encodings, and semantic labels, employing graph attention mechanisms to model spatial constraints among keypoints.

### 3.2.2 Edge Feature-Enhanced Graph Attention

We extend standard GAT by incorporating edge features into attention computation:

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j \| \mathbf{W}_e\mathbf{e}_{ij}]))}{\sum_{k \in \mathcal{N}_i} \exp(\cdot)}$$

where $\mathbf{e}_{ij}$ represents the edge feature vector and $\mathbf{W}_e$ maps edge features to attention space, enabling the model to distinguish between "anatomical connections" and "spatial proximity" relationships.

### 3.2.3 Dynamic Gating Fusion

We design a lightweight gating network to adaptively balance CNN and GAT features:

$$\mathbf{g} = \sigma(\mathbf{W}_g[\text{Proj}(\mathbf{F}_{cnn}) \| \mathbf{F}_{gat}] + \mathbf{b}_g)$$
$$\mathbf{F}_{fusion} = \mathbf{g} \odot \mathbf{F}'_{cnn} + (1-\mathbf{g}) \odot \mathbf{F}_{gat}$$

When image quality deteriorates, the gating network automatically increases GAT feature weights, leveraging spatial constraints to compensate for local feature uncertainty.

## 3.3 Multimodal Clinical Report Generation

### 3.3.1 Multimodal Feature Fusion

We perform Cross-Attention fusion of structured angular parameters derived from keypoint coordinates, 256-dimensional deep features extracted by CNN-GAT, and patient clinical information:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

where $\mathbf{Q}$ represents angular parameter encodings and $\mathbf{K}, \mathbf{V}$ represent visual features, achieving semantic alignment of heterogeneous information.

### 3.3.2 Structured Report Generation

Based on medical pre-trained LLaVA-Med variants, we utilize fused features $\mathbf{F}_{aligned}$ and expert-designed report templates as conditional inputs to generate standardized clinical reports encompassing "measurement results - abnormality descriptions - diagnostic assessments - treatment recommendations." The system overlays angular measurement lines and keypoint annotations on images to enhance interpretability.

**[FIGURE 2 PLACEHOLDER: Example of generated clinical report with keypoint annotations and angle measurements overlaid on X-ray image]**

## 3.4 Loss Function and Training Strategy

We employ a multi-task learning framework integrating keypoint regression loss and medical constraint loss:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{keypoint} + \lambda_2 \mathcal{L}_{medical}$$

where $\mathcal{L}_{medical}$ constrains predicted keypoints to generate clinically reasonable angle measurements, embedding anatomical priors into the learning process.

## 3.5 Data Acquisition and Processing

To authentically reflect grassroots telemedicine scenarios, we collected 800 real user-uploaded pelvic X-ray images from the Xiaohongshu platform, simulating image quality fluctuations in grassroots remote healthcare settings. We employed active learning strategies, utilizing model prediction uncertainty sampling to optimize annotation efficiency with confidence threshold set at 0.7. Data preprocessing includes OCR privacy masking, resolution standardization (512×512), and multi-dimensional augmentation to enhance model generalization capability.

---

## Tables and Performance Metrics

**Table 1: Keypoint Detection Performance Comparison**

| Method | PCK@0.1 | PCK@0.2 | PCK@0.5 | mAP | Parameters |
|--------|---------|---------|---------|-----|------------|
| ResNet-50 | 0.72 | 0.85 | 0.94 | 0.83 | 25.6M |
| HRNet | 0.78 | 0.89 | 0.96 | 0.87 | 32.1M |
| Traditional GAT | 0.75 | 0.86 | 0.95 | 0.85 | 18.3M |
| **CNN-GAT Fusion** | **0.86** | **0.93** | **0.98** | **0.91** | **28.7M** |

**Table 2: Angular Measurement Accuracy**

| Angle Type | Expert Measurement | AI Measurement | MAE | Correlation | Clinical Accuracy |
|------------|-------------------|----------------|-----|-------------|------------------|
| CE Angle | 25.3°±3.2° | 25.1°±3.1° | 0.8° | 0.94 | 92% |
| Sharp Angle | 42.1°±2.8° | 42.3°±2.9° | 1.1° | 0.91 | 89% |
| Tönnis Angle | 8.7°±1.5° | 8.9°±1.6° | 0.6° | 0.96 | 94% |

**Table 3: Ablation Study Results**

| Component Configuration | PCK@0.1 | PCK@0.2 | Angular MAE | Parameters |
|------------------------|---------|---------|-------------|------------|
| CNN Only | 0.72 | 0.85 | 2.3° | 25.6M |
| GAT Only | 0.68 | 0.82 | 2.8° | 18.3M |
| CNN+GAT (No Edge Features) | 0.81 | 0.90 | 1.5° | 28.7M |
| CNN+GAT (No Constraints) | 0.83 | 0.91 | 1.2° | 28.7M |
| Fixed Fusion Weights | 0.84 | 0.92 | 1.0° | 28.7M |
| **Complete Model** | **0.86** | **0.93** | **0.9°** | **28.7M** |

**Table 4: Expert Consistency Evaluation**

| Evaluation Metric | Expert 1 | Expert 2 | Expert 3 | Average |
|------------------|----------|----------|----------|---------|
| Diagnostic Consistency | 0.87 | 0.89 | 0.85 | 0.87 |
| Accuracy | 91% | 93% | 89% | 91% |
| Sensitivity | 88% | 90% | 86% | 88% |
| Specificity | 94% | 96% | 92% | 94% |

---

## Overleaf LaTeX Conversion Notes

For optimal Overleaf formatting:

1. **Figures**: Replace `[FIGURE X PLACEHOLDER: ...]` with proper `\begin{figure}` environments
2. **Tables**: Convert markdown tables to `\begin{table}` with `\begin{tabular}` environments
3. **Equations**: Mathematical expressions are already in LaTeX format
4. **References**: Add proper `\cite{}` commands for citations
5. **Sections**: Use `\section{}` and `\subsection{}` hierarchy
6. **Bold text**: Convert `**text**` to `\textbf{text}`

This translation maintains the technical rigor expected for BIBM while ensuring clinical relevance and accessibility for both computer science and biomedical audiences. The content has been optimized for Nature/Science standards with concise, authoritative language and clear scientific progression.