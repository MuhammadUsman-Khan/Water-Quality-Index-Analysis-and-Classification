# 💧 Water Quality Index Analysis & Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-TPU%20v5e-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![Best Model](https://img.shields.io/badge/Best%20Model-Decision%20Tree%20(Tuned)-blueviolet?style=flat-square)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-96.37%25-blue?style=flat-square)
![F1 Score](https://img.shields.io/badge/Best%20F1%20(minority)-0.838-orange?style=flat-square)

</div>

---

## 📌 Overview

This project builds a **binary classification pipeline** to determine whether a water sample is **safe or unsafe** for consumption, using 20 water chemistry measurements as input features.

The goal is to support **early risk identification and decision-making** for water quality management — providing an automated, data-driven tool that can flag potentially dangerous water samples based on chemical concentration levels.

> **Task:** Given measurements of 20 chemical parameters, classify each water sample as `safe (1)` or `not safe (0)`.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [Kaggle — Water Quality](https://www.kaggle.com/datasets/mssmartypants/water-quality) |
| **File** | `waterQuality1.csv` |
| **Records** | 7,999 samples |
| **Features** | 20 numeric chemical parameters |
| **Target** | `is_safe` — binary {0: not safe, 1: safe} |
| **Class Imbalance** | ~88.6% unsafe / ~11.4% safe |

### Chemical Parameters & Safety Thresholds

| Parameter | Dangerous If Greater Than |
|---|---|
| Aluminium | 2.8 |
| Ammonia | 32.5 |
| Arsenic | 0.01 |
| Barium | 2.0 |
| Cadmium | 0.005 |
| Chloramine | 4.0 |
| Chromium | 0.1 |
| Copper | 1.3 |
| Fluoride | 1.5 |
| Bacteria | 0 |
| Viruses | 0 |
| Lead | 0.015 |
| Nitrates | 10.0 |
| Nitrites | 1.0 |
| Mercury | 0.002 |
| Perchlorate | 56.0 |
| Radium | 5.0 |
| Selenium | 0.5 |
| Silver | 0.1 |
| Uranium | 0.3 |

---

## 🔬 Project Pipeline

```
Raw Data (Kaggle)
      │
      ▼
Data Preparation
  ├── Type conversion (ammonia & is_safe: object → float)
  ├── Null value removal (3 rows dropped after coercion)
  └── Duplicate check (none found) → final shape: 7,996 rows
      │
      ▼
Exploratory Data Analysis
  ├── Descriptive statistics
  ├── Class distribution (imbalanced: 88.6% unsafe / 11.4% safe)
  ├── Feature histograms (distribution of all 20 parameters)
  ├── Boxplots (outlier inspection — outliers retained as real contamination)
  ├── Correlation heatmap (20x20)
  ├── Top 10 features by correlation with target
  ├── Pairplot
  └── Mean chemical values grouped by safety class
      │
      ▼
Data Transformation
  └── StandardScaler applied to all 20 features
      │
      ▼
Train/Test Split
  └── 80/20 stratified split (stratify=y to preserve class ratio)
      Train: 6,396 samples | Test: 1,600 samples
      │
      ▼
Model Training & Evaluation
  ├── Logistic Regression
  ├── Decision Tree           ← Selected as best baseline
  ├── Random Forest
  ├── SVC
  └── K-Nearest Neighbors
      │
      ▼
Hyperparameter Tuning
  └── GridSearchCV on Decision Tree (5-fold CV, F1 scoring, 72 candidates)
      │
      ▼
Final Evaluation & Model Export
  ├── Confusion Matrix
  └── Model saved → best_dt_model.joblib
```

---

## 🧠 Models Trained

| Model | Type |
|---|---|
| Logistic Regression | Linear |
| Decision Tree | Tree-based |
| Random Forest | Ensemble |
| Support Vector Classifier (SVC) | Kernel-based |
| K-Nearest Neighbors | Instance-based |

---

## 📈 Results

### Baseline Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.9087 | 0.7143 | 0.3297 | 0.4511 |
| **Decision Tree** | **0.9581** | **0.8212** | **0.8077** | **0.8144** |
| Random Forest | 0.9575 | 0.9453 | 0.6648 | 0.7806 |
| SVC | 0.9419 | 0.8938 | 0.5549 | 0.6847 |

> ✅ **Decision Tree** was selected as the best baseline model due to its superior F1-score on the minority (safe) class.

### After Hyperparameter Tuning (GridSearchCV)

| Metric | Value |
|---|---|
| **Accuracy** | **96.37%** |
| **Precision** | **85.23%** |
| **Recall** | **82.42%** |
| **F1-Score** | **83.80%** |

**Best Hyperparameters found:**
```python
{
  'criterion':         'entropy',
  'max_depth':         30,
  'min_samples_leaf':  1,
  'min_samples_split': 2
}
```

### Classification Report (Tuned Model)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0.0 (Not Safe) | 0.98 | 0.98 | 0.98 | 1418 |
| 1.0 (Safe) | 0.85 | 0.82 | 0.84 | 182 |
| **Weighted Avg** | **0.96** | **0.96** | **0.96** | **1600** |

---

## 🔑 Key Findings

- **Class Imbalance:** The dataset is heavily imbalanced (~88.6% unsafe), making **stratified sampling** essential and F1-score the preferred evaluation metric over raw accuracy
- **Outliers Retained:** Several parameters show extreme values in boxplots, but these represent genuine contamination events — not noise — so no outlier removal was applied
- **Feature Significance:** Mean chemical concentrations are significantly higher in unsafe samples across most parameters, confirming the predictive power of chemistry measurements
- **Decision Tree vs Random Forest:** While Random Forest achieved higher precision (0.945), it suffered from low recall (0.665) on the minority class — Decision Tree's balanced precision-recall tradeoff made it the better choice
- **Tuning Impact:** GridSearchCV improved the Decision Tree F1-score from **0.8144 → 0.8380**, a meaningful gain on the harder-to-classify safe class

---

## 🛠️ Tech Stack

```python
pandas        # Data loading and manipulation
numpy         # Numerical operations
seaborn       # Heatmaps, boxplots, pairplots, countplots
matplotlib    # Visualizations
missingno     # Missing value visualization
scikit-learn  # Preprocessing, models, GridSearchCV, metrics
joblib        # Model serialization
kagglehub     # Dataset download
```

**Environment:** Google Colab with TPU v5e accelerator

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/MuhammadUsman-Khan/Water-Quality-Index-Analysis-and-Prediction.git
cd Water-Quality-Index-Analysis-and-Prediction
```

### 2. Install Dependencies
```bash
pip install pandas numpy seaborn matplotlib missingno scikit-learn joblib kagglehub
```

### 3. Run the Notebook
Open `Water_Quality_Index_Analysis_&_Regression.ipynb` in Jupyter or Google Colab.

The dataset is automatically downloaded via `kagglehub`:
```python
import kagglehub
path = kagglehub.dataset_download("mssmartypants/water-quality")
```

> ⚠️ A Kaggle account and configured API credentials are required for `kagglehub` to work.

### 4. Load the Saved Model
```python
import joblib
model = joblib.load('best_dt_model.joblib')
predictions = model.predict(X_new)
```

---

## 📁 Repository Structure

```
Water-Quality-Index-Analysis-and-Prediction/
│
├── Water_Quality_Index_Analysis_&_Regression.ipynb   # Main notebook
└── README.md                                          # Project documentation
```

---

## 🔮 Future Improvements

- [ ] Address class imbalance using SMOTE or class-weight adjustments
- [ ] Evaluate XGBoost and LightGBM for potentially higher minority-class recall
- [ ] Add SHAP / LIME explainability to identify which chemicals drive predictions
- [ ] Build a prediction API using FastAPI or Flask
- [ ] Deploy as an interactive web app using Streamlit

---

## 🙋‍♂️ Author

**Muhammad Usman Khan**

[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MuhammadUsman-Khan)



---

<div align="center">
  <i>If you found this project useful, consider giving it a ⭐ — it helps others discover it!</i>
</div>
