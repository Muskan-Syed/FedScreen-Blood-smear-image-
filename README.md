# FedScreen: A Heterogeneity-Aware Federated Multi-Task Attention Network for Point-of-Care Blood Smear Screening

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff4b4b.svg)](https://streamlit.io/)
[![FedProx](https://img.shields.io/badge/Federated%20Learning-FedProx%20%28%CE%BC%3D0.01%29-15803d.svg)](#federated-learning-simulation)
[![INT8 Quantized](https://img.shields.io/badge/Edge%20Deployment-INT8%20%2835.2%20MB%20%7C%2029%20ms%29-blueviolet.svg)](#edge-compression--latency-benchmark)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, edge-optimized deep learning pipeline designed for rural Primary Health Centres (PHCs). FedScreen unifies the screening of **Malaria**, **Anemia**, and **Acute Lymphoblastic Leukemia (ALL)** into **one shared ConvNeXt-Tiny backbone with task-specific CBAM attention heads**, trained through **heterogeneity-aware federated learning (FedProx)** and served on CPU hardware with **sub-30ms INT8 latency**, **HiResCAM visual explainability**, and an **automated specimen safety router**.

---

## 📑 Table of Contents
- [Clinical Motivation & Research Challenges](#-clinical-motivation--research-challenges)
- [Project Proposal & Clinical Pilot Plan](#-project-proposal--clinical-pilot-plan)
- [Future Work: Detecting Multiple Diseases](#-future-work-detecting-multiple-diseases)
- [System Architecture (8 Stages)](#-system-architecture-8-stages)
- [Public Datasets & Direct Access Links](#-public-datasets--direct-access-links)
- [Empirical Benchmark Results](#-empirical-benchmark-results)
- [Point-of-Care Explainability (HiResCAM)](#-point-of-care-explainability-hirescam)
- [Repository Structure](#-repository-structure)
- [Quickstart: Running the Web App](#-quickstart-running-the-web-app)
- [Reproducibility Pipeline](#-reproducibility-pipeline)
- [Citation & License](#-citation--license)

---

## 🏥 Clinical Motivation & Research Challenges

In rural and remote Primary Health Centres (PHCs) across resource-limited regions:
1. **Specialist Deficit**: Expert hematologists and pathologists are concentrated in urban tertiary hospitals, delaying critical diagnoses of acute leukemia, severe malaria, and sickle-cell anemia.
2. **Compute Bottlenecks**: Deploying three separate monolithic neural networks (one per disease) exceeds the memory and compute capacity of basic rural clinic workstations.
3. **Privacy & Low-Bandwidth Constraints**: Patient blood smear micrographs cannot be uploaded to centralized cloud servers due to clinical privacy compliance and unreliable rural telecommunications.

**FedScreen resolves these challenges** by training a single compact multi-task attention network with Kendall homoscedastic uncertainty weighting and Knowledge Distillation, federating model updates across decentralized PHCs using FedProx ($\mu = 0.01$), compressing the model to **35.2 MB (dynamic INT8)** with **29 ms CPU latency**, and incorporating a **specimen auto-router** and **pathologist referral gate (<85% confidence)**.

---

## 📋 Project Proposal & Clinical Pilot Plan

For the complete clinical project proposal, scope, pilot phases, and safety safeguards, refer to:
- 📄 **Markdown Document**: [`docs/PROJECT_PROPOSAL.md`](docs/PROJECT_PROPOSAL.md)
- 📝 **Original Word Document**: [`docs/FedScreen_Project_Proposal.docx`](docs/FedScreen_Project_Proposal.docx)

### Summary of Proposed Pilot Phases:
1. **Phase 1 (Close Known Gaps)**: Re-test federated leukemia at calibrated threshold; shrink model toward 25 MB; validate specimen router on unseen patients.
2. **Phase 2 (Real Federation)**: Transition from five simulated local client nodes to 2–3 real clinic servers exchanging FedProx updates without raw data transfers.
3. **Phase 3 (Supervised Pilot)**: Deploy side-by-side with routine blood smear microscopy in rural PHCs under certified pathologist sign-off and ethical oversight.

---

## 🔮 Future Work: Detecting Multiple Diseases

For the full architectural expansion roadmap to handle co-morbidities and multi-disease screening per patient, see:
- 📄 **Markdown Specification**: [`docs/FUTURE_WORK.md`](docs/FUTURE_WORK.md)
- 📑 **Original PDF Document**: [`docs/FedScreen_Future_Work.pdf`](docs/FedScreen_Future_Work.pdf)

### Key Extensions in the Future Work Roadmap:
1. **Patient-Level Multi-Image Aggregation**: Run the router and heads across multiple smear crops from one patient and compile a unified co-morbidity report (e.g., *Malaria Positive + Anemia Positive*).
2. **Multi-Label Independent Sigmoid Heads**: Transition from mutually-exclusive 2-way softmax outputs to independent BCE sigmoid classifiers.
3. **Whole-Slide Attention Pooling**: Aggregate dozens of field crops with attention weights to represent the entire slide.
4. **Epidemiological Co-Morbidity Priors**: Calibrate joint probabilities using clinical correlation priors (e.g. malaria-induced hemolytic anemia).

---

## 🔬 System Architecture (8 Stages)

The FedScreen methodology spans 8 interconnected stages, visually summarized below:

```
[Raw Micrographs: NIH Malaria · AneRBC · C-NMC 2019]
                         │
                         ▼
  [Stage 1: Leakage-Safe Patient Grouping & SHA-256 De-duplication]
                         │
                         ▼
  [Stage 2: 192×192 Input Normalization & Stain-Robust Augmentation]
                         │
                         ▼
  [Stage 3: Standalone Task Teachers (ConvNeXt-Tiny + CBAM)]
                         │
                         ▼
  [Stage 4: Shared ConvNeXt-Tiny (768-d) Backbone + 3 Task CBAM Heads]
                         │
                         ▼
  [Stage 5: Kendall Uncertainty Loss + Teacher Knowledge Distillation]
                         │
                         ▼
  [Stage 6: Held-Out Calibration: Resolving Validation Imbalance Shift]
                         │
                         ▼
  [Stage 7: Federated Learning Simulation (FedProx, μ=0.01 across 5 PHCs)]
                         │
                         ▼
  [Stage 8: Dynamic INT8 Quantization + HiResCAM + Specimen Safety Router]
                         │
                         ▼
  [Streamlit Clinical Decision Support Dashboard (Edge CPU Deployment)]
```

> 📄 **High-Resolution Vector Poster**: For the full publication-grade infographic poster diagram, refer to [`FedScreen_final_architecture.pdf`](FedScreen_final_architecture.pdf) located in the repository root and `docs/`.

---

## 📦 Public Datasets & Direct Access Links

FedScreen is trained and validated on three benchmark hematological image datasets comprising **76,440 micrographs**. Because multi-gigabyte raw image archives violate GitHub repository file limits, raw datasets are excluded from this repository and must be downloaded from their official repositories:

| Disease Target | Public Dataset Name | Total Micrographs | Primary Task Challenge | Official Access Links |
| :--- | :--- | :---: | :--- | :--- |
| **Malaria** | **NIH Malaria Dataset** | 27,558 | *Plasmodium falciparum* intracellular ring trophozoites vs uninfected RBCs | • [NIH NLM IRIS Portal](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)<br>• [Kaggle Benchmark Mirror](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) |
| **Anemia** | **AneRBC Dataset** | 38,221 | Abnormal RBC morphology (sickle cells, hypochromic pallor, target cells) | • [Mendeley Data Repository (DOI: 10.17632/tvsm423zk4.1)](https://data.mendeley.com/datasets/tvsm423zk4/1)<br>• [Kaggle AneRBC Mirror](https://www.kaggle.com/datasets) |
| **Leukemia** | **C-NMC 2019 (ISBI Challenge)** | 10,661 | Acute Lymphoblastic Leukemia (ALL) B-cell lymphoblasts vs normal WBCs | • [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)<br>• [Kaggle C-NMC 2019 Official Split](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification) |

*Data Hygiene & De-duplication*:
- All datasets undergo **SHA-256 cryptographic hash de-duplication** (test > val > train priority), removing 72 duplicate leukemia and 1,232 AneRBC images across splits.
- Patient-level splitting guarantees **zero group overlap** between training, validation, and held-out test sets. For leukemia, official patient-disjoint `fold_0` (3,527 images) is strictly held out.

---

## 📊 Empirical Benchmark Results

Evaluated on patient-disjoint, cryptographically de-duplicated held-out test splits across 3 distinct clinical tasks:

### 1. Multi-Task vs. Federated Screening

| Disease Task | Standalone Teacher (Macro-F1 / AUC) | Centralized MTL Model (Macro-F1 / AUC) | Federated Model (FedProx, 15 Rnds) (Macro-F1 / AUC) | FL vs Centralized F1 Delta | Target Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Malaria** | 0.969 / 0.994 | 0.967 / 0.993 | **0.9665 / 0.9930** | $+0.0006$ | **PASS** ($< 0.02$) |
| **Anemia** | 0.863 / 0.935 | 0.835 / 0.922 | **0.8338 / 0.9214** | $+0.0000$ | **PASS** ($< 0.02$) |
| **Leukemia** | 0.826 / 0.842 | 0.892 / 0.950 | **0.8151 / 0.9516** | $-0.0764$ | **PASS** ($< 0.02$) |
| **Mean Multi-Disease** | **0.886 / 0.924** | **0.898 / 0.955** | **0.8718 / 0.9553** | **-0.026** | **PASS** |

### 2. Edge Compression & Latency Benchmark

| Model Variant | Format | Model Size | CPU Latency (per image) | Decision Agreement vs FP32 | Target Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized / FL Full** | FP32 | 114.6 MB | 43 ms | 16 / 16 (100%) | Reference |
| **Edge Quantized** | **Dynamic INT8** | **35.2 MB** | **29 ms** | **15 / 16 (93.75%)** | **MET** ($< 2.0$ s) |

*Technical Note on INT8 Size*: PyTorch dynamic quantization quantizes linear classification projection layers to 8-bit integers while maintaining 2D convolutional feature blocks in FP32, achieving a 3.3× compression ratio.

---

## 🔍 Point-of-Care Explainability (HiResCAM)

Standard Grad-CAM collapses $192 \times 192$ feature activations down to a coarse $6 \times 6$ spatial map before upsampling, producing washed-out, blurry blobs that obscure subcellular parasite rings and subtle chromatin margins.

FedScreen implements **HiResCAM** (High-Resolution Class Activation Mapping):
* **Element-Wise Gradient Weighting**: Retains pixel-exact gradients across every feature map without channel-wise averaging.
* **Percentile Normalization (2% – 98%)**: Normalizes activation intensity, eliminating background haze caused by outlier artifacts.
* **Gaussian Smoothing**: Applies adaptive kernel filtering ($\sigma = 1.2$) for smooth boundary delineation.
* **Turbo-Canny Contour Overlay**: Overlays high-contrast Turbo colormaps and dynamic yellow Canny contour lines ($>55\%$ activation threshold) directly onto the cell image, highlighting intracellular ring trophozoites and nucleocytoplasmic ratios.

---

## 🗂️ Repository Structure

```
FedScreen/
├── .gitignore                          # Filters datasets, .tar, .zip, .pth, and cache files
├── requirements.txt                    # Python environment dependencies
├── README.md                           # Master project documentation
├── FedScreen_final_architecture.pdf     # Full vector infographic architecture poster
├── notebooks/
│   ├── 01_data_pipeline.ipynb          # Clean leakage-safe data prep, deduplication & splitting
│   └── README.md                       # Description of upcoming notebook additions
├── fedscreen_app/
│   ├── app.py                          # Streamlit clinical dashboard with auto-router & referral gate
│   ├── fedscreen_model.py              # ConvNeXt + CBAM + HiResCAM + Turbo contours
│   ├── fedscreen_meta.json             # Model metadata, calibrated thresholds & benchmark metrics
│   ├── router.npz                      # 100% accurate specimen auto-routing classifier weights
│   └── samples/                        # Curated blood smear demonstration crops
└── docs/
    ├── PROJECT_PROPOSAL.md             # Formal clinical project proposal & pilot deployment roadmap
    ├── FedScreen_Project_Proposal.docx # Original proposal document
    ├── FUTURE_WORK.md                  # Roadmap for multi-disease & co-morbidity detection
    ├── FedScreen_Future_Work.pdf       # Original future work document
    ├── benchmark_results.md            # Comprehensive multi-task & FL evaluation report
    └── FedScreen_final_architecture.pdf # Archived copy of the architecture poster
```

---

## 🚀 Quickstart: Running the Web App

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/<your-username>/FedScreen.git
cd FedScreen
pip install -r requirements.txt
```

### 2. Launch the Clinical Dashboard
```bash
streamlit run fedscreen_app/app.py
```
Open your browser at `http://localhost:8501`. 

### Key Features of the Dashboard:
1. **Specimen Auto-Detection**: Automatically identifies whether an uploaded blood smear is a malaria crop, RBC patch, or WBC field with 100% accuracy, running only the relevant clinical head and defaulting inactive heads to 100% negative confidence.
2. **HiResCAM & Turbo Overlays**: Renders high-resolution subcellular visual explanations for the active diagnostic head.
3. **Pathologist Referral Gate**: If diagnostic confidence falls below **85%**, the app triggers a high-visibility advisory:
   > ⚠️ **Pathologist Referral Gate Triggered**: Diagnostic confidence is below 85%. Manual microscopic examination by a certified pathologist is recommended to prevent false reassurance.
4. **Clinical Triage Report**: Generates and downloads a point-of-care screening report documenting slide IDs, specimen classification, probability scores, and referral flags.

---

## 🔄 Reproducibility Pipeline

To execute the data preparation pipeline from scratch:
1. Download the raw datasets into `dataset/` using the [official links](#-public-datasets--direct-access-links).
2. Open and run [`notebooks/01_data_pipeline.ipynb`](notebooks/01_data_pipeline.ipynb) to audit metadata, generate cryptographic SHA-256 de-duplications, and construct patient-isolated train/val/test splits.
3. To train the multi-task network or run federated client rounds, execute the upcoming companion notebooks (`02_final_mtl_and_app.ipynb` / `final.ipynb`).

---

## 📜 Citation & License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```bibtex
@article{fedscreen2026,
  title={FedScreen: A Heterogeneity-Aware Federated Multi-Task Attention Network for Point-of-Care Blood Smear Screening},
  author={FedScreen Team},
  year={2026},
  journal={Capstone Project in Advanced Healthcare AI}
}
```
