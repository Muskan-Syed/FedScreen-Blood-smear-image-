# FedScreen Pipeline Notebooks

This directory contains the end-to-end experimental and training notebooks for the **FedScreen** multi-task blood-cell screening framework.

### Notebook Organization

1. **`01_data_pipeline.ipynb`** *(Included in this repository)*
   - **Audit & Ingestion**: Ingests raw public datasets (NIH Malaria, AneRBC, C-NMC 2019).
   - **Leakage-Safe Splitting**: Implements patient-level isolation (`patient P<n>` for Malaria, parent image grouping for AneRBC, official patient-disjoint `fold_0` for Leukemia) ensuring **zero group leakage** across train/val/test splits.
   - **Cryptographic De-duplication**: SHA-256 hash de-duplication with test > val > train priority (eliminating 72 duplicate leukemia and 1,232 AneRBC images).
   - **Stain-Robust Preprocessing**: Standardizes input dimensions to $192 \times 192 \times 3$ with per-image per-channel $z$-score normalization and color-jitter/MixUp augmentations.

2. **`02_final_mtl_and_app.ipynb`** *(Awaiting final refresh)*
   - Multi-Task Attention Network training with Kendall uncertainty weighting and teacher Knowledge Distillation ($\alpha = 0.30$).
   - Heterogeneity-aware federated learning simulation (FedProx, $\mu = 0.01$) across 5 non-IID virtual Primary Health Centres.
   - Dynamic INT8 quantization and specimen auto-router fitting.

3. **`final.ipynb`** *(Awaiting final refresh)*
   - Master 51-cell conference-grade reproducible pipeline integrating all 23 pipeline stages from raw data ingestion to Cloudflare tunnel edge deployment.

> **Note**: Notebooks `02_final_mtl_and_app.ipynb` and `final.ipynb` are currently being refreshed in Google Colab with updated Federated INT8 artifacts and HiResCAM visual overlays, and will be committed directly once execution completes.
