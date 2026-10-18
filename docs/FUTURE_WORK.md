# FedScreen — Future Work: Detecting More Than One Disease

**Date:** Oct 17, 2026 · `@syed`  
**PDF Document Source:** [`FedScreen_Future_Work.pdf`](file:///C:/Users/DELL/OneDrive/Documents/GitHub/FedScreen/docs/FedScreen_Future_Work.pdf)

---

## Where We Are Today
FedScreen checks one disease per image, and it does that well.

- **One shared ConvNeXt-Tiny backbone** feeds three small heads: malaria, anemia and leukemia.
- **A specimen router** recognises the image type and runs only the matching head.
- **The other two diseases** are shown as negative, because they were not tested on that image.

---

## Why It Cannot Say "Two Diseases" Yet
The model has never seen a patient with more than one condition:

- **Single-Specimen Datasets:** Each dataset is a different specimen (single malaria cell crop, red-cell field, white-cell crop), and every image has exactly one disease label.
- **Specialized Cell Focus:** Each head only learned its own cell type, so it cannot judge the others.
- **1-Head Gating:** The router picks one head, so by design only one answer comes out.

---

## Roadmap to Detect 2 or 3 Diseases Together

### 1. Patient-Level Report (No New Data Needed)
One patient typically has multiple slide images. Run the router and the matching head on each image, then aggregate and combine the results into a unified patient-level triage report — for example, *"Patient #104: Malaria Positive, Anemia Positive"*.

### 2. Multi-Label Training
Replace the two-way softmax with independent yes/no (sigmoid) outputs per disease, trained with Binary Cross-Entropy (BCE). This allows several diagnostic outputs to be flagged as positive simultaneously on complex multi-cell smear fields.

### 3. Slide-Level Model (Attention Pooling)
Let the neural network process dozens of cell crops from an entire whole-slide image (WSI) and weigh the clinically significant cells using **attention pooling**, producing one consolidated set of disease probabilities per patient slide.

### 4. Co-Morbidity Correlation Modeling
Certain hematological conditions frequently co-occur (e.g., severe *P. falciparum* malaria causing secondary hemolytic anemia). Calibrate joint probabilities and tune per-disease operating cut-offs so combined diagnostic outputs reflect clinical co-morbidity priors.

---

## What We Need and How We Will Measure It

- **Clinical Co-Morbidity Data:** Real-world patient slides with confirmed multiple diagnoses and multiple smear fields per patient, collected from clinical partner hospitals under Institutional Ethics Committee (IEC) approval. (Note: Synthetic mixing of crops from different patients is scientifically invalid).
- **Patient-Isolated Splits:** Strict patient-level grouping ensuring all slides/images from any individual patient remain entirely within train, validation, or test sets.
- **Evaluation Metrics:** Per-disease AUC and Macro-F1, exact-match subset accuracy (subset accuracy for the full set of conditions), and joint sensitivity for high-risk combinations (e.g., Malaria + Anemia).
- **Federated Co-Learning:** Preserve the FedProx ($\mu = 0.01$) decentralized aggregation framework, allowing rural PHCs with varied epidemiological disease distributions to train locally without transferring raw micrographs.

---

## Safety Note

> [!IMPORTANT]
> **FedScreen is an edge-optimized screening decision-support aid, not a diagnostic device.** Every positive or borderline finding requires pathologist verification. When diagnostic confidence falls below the calibrated referral threshold (<85%), the system flags *"Refer to Pathologist"* rather than offering false reassurance.
