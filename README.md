# 🔬 3-Tier Hierarchical Skin Cancer Screening & Triage Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://skin-cancer-screening-susith.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, edge-deployable clinical AI pipeline designed to screen dermoscopic lesions, halt out-of-distribution (OOD) artifacts, and triage suspicious lesions through a 3-tier cascaded architecture.

---

## 📌 Clinical Context & Overview

Standard monolithic deep-learning classifiers trained on curated dermoscopic datasets (e.g., ISIC) often suffer from severe failure modes when exposed to real-world clinical intake:
1. **Hallucination on Non-Lesions:** Flat normal skin, anatomical background, or non-biological surfaces (e.g., examination tables, wood grain) are forcibly classified into one of the trained disease bins.
2. **Lineage Conflation:** Subtle amelanotic or non-pigmented carcinomas are frequently masked by melanocytic mimics.
3. **Symmetric Risk Penalty:** Standard softmax cross-entropy penalizes all classification errors equally, whereas a false negative on an invasive carcinoma carries a vastly higher clinical consequence than a false positive on a benign mole.

To solve these challenges, this system implements a **3-Tier Cascaded Screening Pipeline** combining spatial-frequency artifact rejection, histogenetic lineage decoupling, and asymmetric risk calibration.

---

## 🏗️ System Architecture

[ Input Dermoscopic Specimen ]
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 1: Adaptive Saliency & Specimen Gatekeeper             │
│ • Local Contrast Gradient (Center vs. Immediate Ring)       │
│ • Directional Anisotropy (Sobel Gradient Axis Ratio)        │
│ • MobileNetV3-Small Lesion vs. OOD Binary Classifier        │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   [ Saliency/OOD Failed ]               [ Specimen Validated ]
   ⚠️ HALT & REJECT                       │
   (Prevents cutis/desk hallucination)    ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 2: Dual-Head Hierarchical Screener (EfficientNet-B0)   │
│ • Substrate Cleanup: DullRazor Black-Hat Inpainting         │
│ • Color Normalization: Shades of Gray Constancy (p=6)       │
│ • Head 1: Histogenetic Lineage Decoupling (3 Classes)       │
│ • Head 2: Terminal Pathology Classification (6 Classes)     │
│ • Visual Attribution: Backpropagated Grad-CAM Engine        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 3: Asymmetric Decision Matrix & Ambiguity Engine       │
│ • Normalized Shannon Entropy Ambiguity Scoring              │
│ • Asymmetric Malignancy Escalation (Melanoma & BCC)        │
│ • Keratotic Tail Logit Override (Bowen's / AKIEC Catch)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
            [ Actionable Clinical Triage Output ]
    🔴 HIGH RISK │ 🟡 PRE-MALIGNANT │ 🟢 BENIGN │ ⚠️ UNCERTAIN


## ⚙️ Detailed Pipeline Mechanics

### Tier 1: Intake Saliency Gate & Artifact Halting
Before passing specimens to the deep neural backbone, inputs undergo non-biological and featureless substrate filtration:
* **Radial Local Contrast:** Calculates luminance differential between the central core ($35\% \text{ to } 65\%$) and the immediate pericentral annular ring ($20\% \text{ to } 80\%$). Eliminates flat, non-lesional cutis and uniform desk surfaces.
* **Directional Anisotropy:** Computes orthogonal Sobel gradients ($G_x$ vs. $G_y$) to identify high-frequency linear grain patterns, successfully eliminating directional wood grain and background artifacts.
* **Gatekeeper CNN:** Lightweight MobileNetV3-Small ensures only valid biological lesions enter the compute pipeline.

### Tier 2: Dual-Head Screener & Visual Explainability
* **Substrate Preprocessing:** Integrated morphological hair removal (DullRazor using black-hat structural closures) and Shades-of-Gray color constancy normalize lighting and remove obstructive occlusions.
* **Histogenetic Lineage Branching:** Features extracted from the EfficientNet-B0 backbone branch into:
  * **Stage 1 Lineage:** Melanocytic, Keratinocytic/Epidermal, or Vascular.
  * **Stage 2 Histology:** Melanoma (`mel`), Basal Cell Carcinoma (`bcc`), Actinic Keratosis / Bowen's (`akiec`), Melanocytic Nevus (`nev`), Seborrheic Keratosis (`kerat`), Vascular (`vasc`).
* **Explainable AI (Grad-CAM):** Extracts feature attribution from the final convolutional stage (`features[-1]`) to verify that activations center on lesion morphology rather than peripheral artifacts.

### Tier 3: Asymmetric Decision Matrix
* **Entropy-Based Ambiguity Index:** Calculates normalized Shannon Entropy $H(X) / \ln(N)$. If ambiguity exceeds $65\%$, the specimen is escalated to `⚠️ HIGH UNCERTAINTY / INDETERMINATE MORPHOLOGY` rather than yielding an unconfident guess.
* **Asymmetric Risk Scaling:** Lowers thresholds for high-consequence pathologies:
  * Suspected Malignancy triggered if $P(\text{mel}) \ge 0.42$ or $P(\text{mel}) + P(\text{bcc}) \ge 0.48$.
  * BCC triggered at $P(\text{bcc}) \ge 0.38$.
  * Pre-malignant Bowenoid override captures atypical keratinocytic tails ($P(\text{akiec}) \ge 0.015$ with elevated keratinocytic lineage).

---

## 📊 Validation & Audit Suite

Stress-tested against a diverse 17-specimen challenge battery comprising non-lesion surfaces, look-alike mimics, and biopsy-verified pathologies:

| Specimen Class | Key Morphological Metrics | Tier 1 Gate | Diagnostic Outcome | Clinical Triage Decision |
| :--- | :--- | :--- | :--- | :--- |
| **Desk Surface** | Contrast: `1.49` ($< 4.65$) | **REJECTED** | Specimen Rejected | ⚠️ **Specimen Rejected (OOD)** |
| **Normal Cutis** | Contrast: `4.58` ($< 4.65$) | **REJECTED** | Specimen Rejected | ⚠️ **Specimen Rejected (Cutis)** |
| **Wood Grain** | Anisotropy: `0.610` ($> 0.280$) | **REJECTED** | Specimen Rejected | ⚠️ **Specimen Rejected (Artifact)** |
| **Nodular BCC** | Delta Lum: `19.49` ($\ge 18.0$) | **PASSED** | Basal Cell Carcinoma | 🔴 **HIGH RISK (Malignancy)** |
| **Invasive SCC** | Contrast: `29.11`, Delta: `50.69` | **PASSED** | Basal Cell Carcinoma | 🔴 **HIGH RISK (Malignancy)** |
| **Bowen's Disease** | AKIEC Tail Trigger Active | **PASSED** | Melanocytic Nevus (Top Logit) | 🟡 **PRE-MALIGNANT (AK/Bowen's)** |
| **Seborrheic Keratosis**| Contrast: `46.50` | **PASSED** | Keratotic Mimic | 🟢 **BENIGN KERATOSIS** |
| **Spitz Nevus** | Contrast: `68.76` | **PASSED** | Melanocytic Nevus | 🟢 **BENIGN MELANOCYTIC NEVUS** |

---

## 🚀 Quickstart & Local Installation

### Prerequisites
* Python 3.10+
* Git LFS installed (`git lfs install`)

### Setup Instructions

# 1. Clone the repository
git clone https://github.com/susith-athukorala/skin-cancer-screening-pipeline.git
cd skin-cancer-screening-pipeline

# 2. Create and activate a virtual environment
# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt / PowerShell):
# python -m venv venv
# venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the application locally
streamlit run streamlit_app.py

### ⚖️ Clinical & Regulatory Disclaimer

**FOR INVESTIGATIONAL, RESEARCH, AND EDUCATIONAL USE ONLY.**

This software, including its underlying machine learning models, saliency gates, and triage algorithms, is provided strictly for academic research, algorithm validation, and educational demonstrations in digital health and biomedical informatics.

* **Not a Certified Medical Device:** This tool has NOT been evaluated, cleared, or approved by the Australian Therapeutic Goods Administration (TGA), the U.S. Food and Drug Administration (FDA), or any other national or international regulatory authority as Software as a Medical Device (SaMD).
* **No Diagnostic Reliance:** The outputs, risk stratifications, and triage suggestions generated by this pipeline do not constitute formal medical diagnoses, clinical prognoses, or definitive treatment directives. They must **never** be used as a standalone determinant or substitute for professional medical judgment.
* **Clinical Practice Standards:** Accurate dermatological assessment requires comprehensive patient history, total-body skin examination, in-person clinical dermoscopy by a qualified medical practitioner, and, where indicated, histological tissue biopsy.
* **Limitation of Liability:** The authors, contributors, and affiliated institutions assume no legal responsibility or liability for any clinical decisions, adverse outcomes, delays in care, or damages resulting directly or indirectly from the use, misuse, or interpretation of this software or its outputs.
