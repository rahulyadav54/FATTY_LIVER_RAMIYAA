# Fatty Liver Classification Using Siamese Neural Networks (ResNet-50)

This repository provides an end-to-end deep learning framework for detecting and classifying fatty liver disease (steatosis) from B-mode ultrasound images using a **Siamese Neural Network (SNN)** with pre-trained **ResNet-50** backbone and self-supervised **contrastive pre-training**.

### 🌐 Live Production Deployments
- **Official Custom Domain**: [https://livercare.zayacodehub.in](https://livercare.zayacodehub.in)
- **Vercel Frontend**: [https://fatty-liver-gamma.vercel.app](https://fatty-liver-gamma.vercel.app)
- **Render Backend API**: [https://fatty-liver-ramiyaa.onrender.com](https://fatty-liver-ramiyaa.onrender.com)
- **Copy-Paste Deployment & Domain Setup Guide**: See [DEPLOYMENT_CONFIG_README.md](file:///c:/PROJECTS/FATTY%20LIVER%20DETECTION%20SYSTEM/FATTY_LIVER_RAMIYAA/DEPLOYMENT_CONFIG_README.md) for all environment variables, DNS records, and Supabase settings.

---

## 📌 Executive Summary & Audit Findings

### 1. Verified Dataset Size
- **Primary Source**: IJCARS B-mode Fatty Liver Ultrasound Dataset (`dataset_liver_bmodes_steatosis_assessment_IJCARS.mat`).
- **Patient Cohorts**: 55 unique patient acquisitions (50 Steatosis cohort + 5 Fibrosis cohort).
- **Total Ultrasound Frames**: **550 raw B-mode ultrasound images** (10 frames per patient).
- **Class Breakdown**:
  - `Normal` (Grade 0, < 5% fat): 2 patients (20 frames)
  - `Grade-I` (Mild, 5–35% fat): 23 patients (230 frames)
  - `Grade-II` (Moderate, 35–65% fat): 16 patients (160 frames)
  - `Grade-III` (Severe, > 65% fat): 13 patients (130 frames)
  - `CLD` (Chronic Liver Disease / Cirrhosis): 1 patient (10 frames)

### 2. Leakage-Resistant Patient-Level Evaluation Methodology
- **The Issue**: Naive random splitting across individual frames causes severe patient frame leakage (frames from the same patient appearing in both train and test sets), yielding artificial 100% test accuracy.
- **The Fix**: Implemented **Patient-Level Group Splits** (`GroupShuffleSplit`). All 10 ultrasound frames for any given patient remain strictly in a single split (Train, Val, or Test).

### 3. Rigorous Verified Performance Metrics
- **Binary Classification (Normal vs Abnormal)**: **100.00% Accuracy**
- **Leakage-Free Multi-Class Accuracy**: **36.00% Accuracy**
- **Multi-Class ROC-AUC**: **0.6239**

---

## 🛠️ Installation & Setup

1. **Clone Repository & Navigate to Directory**:
   ```bash
   cd c:\Users\24aiml52\Desktop\FattyLiver\FATTY_LIVER_RAMIYAA
   ```

2. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Execution & Command-Line Usage

### A. Launch Streamlit Medical AI Dashboard
```bash
python -m streamlit run app_streamlit.py
```
*Access in browser at:* `http://localhost:8501`

### B. Run Single-Image Terminal Inference (`infer.py`)
- **Infer from image file**:
  ```bash
  python infer.py --image data/CLD/CLD_000.jpg --binary
  ```
- **Infer from dataset patient acquisition**:
  ```bash
  python infer.py --mat_sample 0 --binary
  ```

### C. Run Leakage-Free Dataset Evaluation
```bash
python scripts/evaluate.py
```

### D. Train the Model Pipeline
```bash
python scripts/train.py --epochs 20 --contrastive_epochs 3
```

---

## 🏗️ Architecture Specifications

- **Encoder Backbone**: Modified ResNet-50 (pre-trained on ImageNet).
- **Projection Head**: Linear(2048 → 512) → ReLU → Linear(512 → 128).
- **Classifier Head**: Linear(2048 → 5 classes).
- **Loss Functions**:
  - **Pre-Training**: NT-Xent (Normalized Temperature-Scaled Cross Entropy) Loss ($T = 0.5$).
  - **Classification**: Weighted Cross-Entropy Loss to handle steatosis class imbalance.

---

## ⚠️ Clinical Research Disclaimer

This application and code are **experimental research prototypes** intended strictly for academic evaluation and clinical research support. It is **not a certified diagnostic device** and must not be used for direct patient diagnosis or medical decision-making. All findings must be evaluated by a qualified radiologist or physician.
