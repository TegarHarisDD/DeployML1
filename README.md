## Obesity Classification Web App

Welcome to the **Obesity Classification** Streamlit application! This repository contains everything you need to explore, predict, and understand obesity categories based on lifestyle and demographic data using three machine learning models: K‑Nearest Neighbors (KNN), Support Vector Machine (SVM), and XGBoost (XGB).

## 🔍 Project Overview

This application allows users to:

* **Explore** dataset metadata and model performance.
* **Input** individual profiles to receive obesity‐category predictions from three tuned models.
* **Bulk‐predict** on CSV files using the XGBoost model and download results.

Our goal is to provide an intuitive interface for non‑technical users to leverage machine learning in assessing obesity levels, fostering awareness of lifestyle impacts on health.

---

## ✨ Features

1. **Three‑Page Streamlit Interface**

   * **About**: Detailed dataset overview, data‑cleaning steps, feature engineering, and tuned model accuracies.
   * **Single Prediction**: Enter personal and lifestyle information via human‑readable forms; view predictions from KNN, SVM, and XGBoost in category labels.
   * **Bulk Prediction**: Upload a CSV of multiple records and download a new CSV with predicted obesity categories (using XGBoost).

2. **Robust Preprocessing Pipeline**

   * Missing‐value standardization, outlier removal (hard bounds + IQR), BMI feature engineering.
   * Label encoding for target and one‑hot encoding for categorical features.
   * Standard scaling of numeric features.
   * SMOTE for class balancing in training.

3. **Multiple Models with Hyperparameter Tuning**

   * KNN, SVM, XGBoost: All tuned via `GridSearchCV` for optimal performance.
   * Persisted scaler, feature list, label encoder, and models for reproducibility.

---

## 🗄️ Data Description

* **Samples**: 2,111
* **Features** (17):

  * Demographics: `Age`, `Gender`, `Height`, `Weight`
  * Dietary & Lifestyle: `CALC`, `FAVC`, `FCVC`, `NCP`, `SCC`, `SMOKE`, `CH2O`, `family_history_with_overweight`, `FAF`, `TUE`, `CAEC`, `MTRANS`
  * **Engineered**: `BMI`
* **Target**: `NObeyesdad` — seven obesity categories from `Insufficient_Weight` to `Obesity_Type_III`.

**Key Cleaning Steps**

* Standardized “?” to NaN, coerced numerics, dropped duplicates & nulls.
* Hard bounds:

  * `Age` ∈ \[14, 80]
  * `Height` ∈ \[1.2 m, 2.2 m]
  * `Weight` ∈ \[30 kg, 200 kg]
* IQR filtering on numeric lifestyle features.

---

## 🛠️ Modeling Pipeline

1. **Preprocessing**

   * Numeric conversion → Null drop → Outlier filtering → BMI computation
   * Label encode target → One‑hot encode categoricals → Train/test split (80/20 stratified)
   * Standard scaling → SMOTE oversampling on training set

2. **Model Training**

   * **Default**: Trained KNN, SVM, XGB with out‑of‑the‑box hyperparameters.
   * **Tuned**: Grid search (5‑fold CV) over relevant hyperparameter grids for each model.

3. **Evaluation**

   * Classification reports (precision, recall, F1‑score) on test set.
   * Accuracy summary persisted for “About” page.

---

## 🧩 App Structure

* **`app.py`**
  Main Streamlit script:

  * Loads artifacts: scaler, `feature_columns`, models, label encoder, accuracies, dataset info.
  * Defines `preprocess_input()` to align incoming data with training pipeline.
  * Renders three pages with forms, tables, and download buttons.

* **Artifacts**

  * `scaler.pkl` — StandardScaler
  * `feature_columns.pkl` — Ordered list of training features
  * `label_encoder.pkl` — Target encoder
  * `knn_tuned.pkl`, `svm_tuned.pkl`, `xgb_tuned.pkl` — Tuned models
  * `accuracy_tuned.pkl` — Dict of model accuracies
  * `dataset_info.md` — Dataset overview & cleaning summary

* **`requirements.txt`**
  Lists all Python dependencies.

---

## 🚀 Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your‑username>/obesity‑classification‑app.git
   cd obesity-classification-app
   ```

2. **Create & activate virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

* Open the provided local URL (e.g. `http://localhost:8501`) in your browser.
* Navigate via the sidebar to explore, predict individually, or run bulk predictions.

---

## 🌐 Deployment

To deploy on **Streamlit Cloud** or any other hosting:

1. Push all files (including `.pkl` artifacts and `dataset_info.md`) to your GitHub repo.
2. Connect your repo in Streamlit Cloud.
3. Configure “Main file” as `app.py`.
4. Deploy!

---

## 📁 Repository Structure

```
├── app.py
├── dataset_info.md
├── requirements.txt
├── scaler.pkl
├── feature_columns.pkl
├── label_encoder.pkl
├── knn_tuned.pkl
├── svm_tuned.pkl
├── xgb_tuned.pkl
├── accuracy_tuned.pkl
└── README.md
```
