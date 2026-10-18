# FedScreen — Project Proposal

**Date:** Oct 17, 2026 · `@syed`  
**Document Source:** [`FedScreen_Project_Proposal.docx`](file:///C:/Users/DELL/OneDrive/Documents/GitHub/FedScreen/docs/FedScreen_Project_Proposal.docx)

---

## Summary
**FedScreen** is one lightweight AI model that helps screen blood-cell images for malaria, anemia and leukemia, and is built so clinics can improve it together without ever sharing patient images. On held-out, patient-disjoint test sets it reaches macro-F1 of **0.967 for malaria**, **0.835 for anemia**, and **0.892 for leukemia**, and a working clinician-facing app already runs on an ordinary CPU in about **29 ms per image**. 

We are asking for support to move from a validated prototype to a small, supervised pilot at primary health centres (PHCs).

---

## The Problem We Want to Ease
In many primary health centres, a blood smear is read by one overstretched technician, or sent away and waited on for days. Three things make it hard to fix with AI:

1. **Many diseases, few resources:** Separate models for each disease are costly to run and maintain on clinic hardware.
2. **Privacy:** Patient images should not leave the clinic, so a single central dataset is often not possible.
3. **Trust:** Models that look excellent in a lab can quietly fail on a new clinic's stain and scanner, and clinicians need to see why a result was given.

---

## Our Approach
FedScreen trains one shared **ConvNeXt-Tiny backbone** (28.6 M parameters) with three small attention heads, one per disease. A router first recognises the specimen type, so only the matching head runs.

- **Clean, leakage-safe data:** Three public datasets (~76,000 cell images) split strictly by patient, de-duplicated by perceptual image hash, and evaluated on an unseen laboratory for leukemia (C-NMC 30-donor deduplicated fold).
- **Specialist teachers:** One model per disease is trained first, with stain-robust augmentation.
- **Joint model:** The shared model learns from all three through uncertainty-weighted loss and knowledge distillation, so no disease drowns out the others.
- **Federated training:** Clinics train locally and share only model updates (FedProx), never patient images.
- **Edge app:** INT8 compression, HiResCAM / Grad-CAM++ heat-map explanations with Canny contours, and a confidence gate that refers uncertain cases to a pathologist.

---

## What Exists Today
Results on held-out, patient-disjoint test data (leukemia: an unseen laboratory). The federated run simulated five virtual health centres.

| Disease | Centralized F1 | AUC | Federated F1 | What we learned |
| :--- | :---: | :---: | :---: | :--- |
| **Malaria** | 0.967 | 0.993 | 0.967 | Federation cost nothing |
| **Anemia** | 0.835 | 0.922 | 0.834 | Hardest task; sharing one backbone cost 0.028 F1 versus its specialist |
| **Leukemia** | 0.892 | 0.950 | 0.815 | Largest federated gap; needs re-checking at the same tuned threshold |

The prototype app uploads an image, identifies the specimen, shows the verdict with an attention heat-map, and refers anything below 85% confidence to a pathologist. The model compresses from 114.6 MB to 35.2 MB and runs in 29 ms per image on CPU.

---

## Plan, Honest Limits and Safe Use

> [!IMPORTANT]
> **FedScreen is a screening aid for trained staff, not a diagnostic device.** Every positive or low-confidence result goes to a pathologist. Timelines and partners below are proposed and to be agreed.

| Phase | Goal | Done When |
| :--- | :--- | :--- |
| **1. Close known gaps** | Re-test federated leukemia at the fixed 0.50 threshold; shrink the model toward the 25 MB target; validate the router on unseen patients | Federated mean F1 drop is under 0.02; size target met |
| **2. Real federation** | Move from five simulated centres to two or three real clinic servers | Updates aggregate without raw images leaving any site |
| **3. Supervised pilot** | Run the app beside routine smear reading, with pathologist sign-off and ethics approval | Agreement with pathologists measured on real, local slides |

### Limits We Are Open About
- **Size Target:** The 25 MB size target is not yet met (currently 35.2 MB because dynamic quantization applies to `nn.Linear` while CBAM convs remain FP32).
- **Federated Leukemia Evaluation:** The leukemia federated result is indicative until re-evaluated under patient-disjoint fold_0.
- **Specimen Router Split:** The specimen router scored 100% on a random split that was not patient-disjoint (primarily serves as an automated safety gate).
- **Public Data:** All data so far is public, so performance at a new clinic is unproven until field validation.
- **Disciplined Iteration:** Early mistakes (data leakage, a failed threshold, collapsed multi-task training) were found, documented and fixed, and that same discipline will guide the pilot.
