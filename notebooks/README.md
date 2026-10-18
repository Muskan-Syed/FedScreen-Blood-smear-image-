# FedScreen Pipeline Notebooks

This directory contains the end-to-end experimental, training, and deployment notebooks for the **FedScreen** multi-task blood-cell screening framework.

### Notebook Organization

1. **[`01_data_pipeline.ipynb`](01_data_pipeline.ipynb)**
   - **Audit & Ingestion**: Ingests raw public datasets (NIH Malaria, AneRBC, C-NMC 2019).
   - **Leakage-Safe Splitting**: Implements patient-level isolation (`patient P<n>` for Malaria, parent image grouping for AneRBC, official patient-disjoint `fold_0` for Leukemia) ensuring **zero group leakage** across train/val/test splits.
   - **Cryptographic De-duplication**: SHA-256 hash de-duplication with test > val > train priority (eliminating 72 duplicate leukemia and 1,232 AneRBC images).
   - **Stain-Robust Preprocessing**: Standardizes input dimensions to $192 \times 192 \times 3$ with per-image per-channel $z$-score normalization and color-jitter/MixUp augmentations.

2. **[`02_final_mtl_and_app.ipynb`](02_final_mtl_and_app.ipynb)**
   - **Multi-Task Attention Network Training**: Kendall homoscedastic uncertainty weighting and teacher Knowledge Distillation ($\alpha = 0.30$) with shared ConvNeXt-Tiny (768-d) backbone and 3 CBAM heads.
   - **Heterogeneity-Aware Federated Learning**: Simulates FedProx ($\mu = 0.01$) across 5 non-IID virtual Primary Health Centres for 15 rounds.
   - **Edge Optimization & Deployment**: Dynamic INT8 quantization (35.2 MB, 29 ms CPU latency), specimen auto-router fitting on the deployed federated backbone, and Cloudflare quick tunnel deployment.
   - **Consolidated Benchmark Table**: Standalone vs. Centralized vs. Federated ablation metrics.

3. **[`final.ipynb`](final.ipynb)**
   - **Master 50-Stage Reproducible Pipeline**: Complete, unified end-to-end notebook connecting all 24 pipeline stages: from raw data acquisition, watershed/Macenko checks, standalone specialist teachers, multi-task joint optimization, federated client simulation, edge INT8 quantization, specimen routing, HiResCAM explainability, and the live clinical triage dashboard.
