# Hospital Readmission Prediction & Power BI Dashboard

---

## Project Overview

This project predicts hospital readmission likelihood for diabetic patients using **Machine Learning (Random Forest Classifier)** and visualizes insights in an **interactive Power BI dashboard**.  
It integrates **data preprocessing, model training, feature interpretation (SHAP), and business intelligence visualization** to support healthcare analytics and patient care improvement.

---

## About the Dataset

- **Dataset name:** `hospital_readmissions.csv`  
- **Source:** Publicly available diabetes hospital readmission dataset [Kaggle](https://www.kaggle.com/datasets/dubradave/hospital-readmissions).  
- **Rows:** ~25,000 hospital records  
- **Columns include:**
  - `age`, `medical_specialty`, `diag_1`, `diag_2`, `diag_3` (diagnoses)
  - `glucose_test`, `a1ctest`, `change`, `diabetes_med`
  - `time_in_hospital`, `n_procedures`, `n_lab_procedures`, `n_medications`
  - `n_outpatient`, `n_inpatient`, `n_emergency`
  - `readmitted` → Target column (“Yes” or “No”)  

The dataset is cleaned, standardized, and transformed for predictive modeling.

---
## Objectives

- Predict the probability of **patient readmission within 30 days**.  
- Identify **key clinical and demographic factors** influencing readmission.  
- Generate a **Power BI-ready dataset** for interactive visual insights.  
- Help hospitals **optimize care and reduce readmission rates**.

---

## Workflow

### 1 Data Preparation
- Loaded dataset: `hospital_readmissions.csv`
- Cleaned and standardized column names.
- Created binary target variable: `readmit_30` (1 = readmitted, 0 = not readmitted).

### 2 Exploratory Data Analysis (EDA)
- Readmission rate distribution.
- Average hospital stay vs. readmission.
- Readmission by medical specialty, diagnosis, diabetes medication, and A1C test.

### 3 Model Pipeline
- **Feature Engineering:** Encoded categorical features and scaled numerics.  
- **Model Used:** Random Forest Classifier  
- **Optimization:** GridSearchCV (Hyperparameter tuning)
- **Evaluation Metrics:**
  - ROC AUC  
  - PR AUC  
  - Accuracy  
  - Precision, Recall, F1-score 

### 4 Feature Explainability (SHAP)
- Used **SHAP (SHapley Additive exPlanations)** to interpret feature importance.
- Extracted top 3 most impactful features for each prediction.
- Exported summary plots (`shap_summary.png`) and feature importance table (`feature_importance_top40.csv`).

### 5 Power BI Dashboard Integration
- Created **dashboard_table.csv** with columns:
  - `top1, top1_val, top2, top2_val, top3, top3_val, pred_readmit_prob, actual_readmit`
- Designed Power BI visuals to analyze model outcomes and key features.

---

## Repository Structure

```

Hospital-Readmission-Prediction
│
├── hospital_readmission_analysis.ipynb   # Main ML pipeline code
├── Hospital_Readmission_Dashboard.pbix   # Power BI dashboard file
├── hospital_readmissions.csv             # Raw dataset 
├── shap_summary.png                   # SHAP feature importance visualization
├── feature_importance_top40.csv       # Top features by model importance
├── dashboard_table.csv                # Power BI-ready dataset
├── readmission_by_specialty.csv       # Specialty-level summary
├── readmission_by_a1c.csv             # A1C test summary
├── readmission_by_medication.csv      # Diabetes medication summary
├── readmission_by_diag.csv            # Diagnosis summary
├── best_rf_readmit_corrected.pkl      # Trained Random Forest model
└── README.md                          # Project documentation

````

---

## Power BI Dashboard Visuals

### Page 1 – Overview
- **KPI Cards:**  
  - Observed Readmit Rate (%)  
  - Avg Predicted Risk (%)  
  - Total Admissions  
  - Predicted High-Risk Count  
- **Risk Distribution** (Clustered Column Chart)  
- **Actual vs Predicted Readmissions** (Stacked Column Chart)  

### Page 2 - SHAP Table
- Top SHAP Features Table

### Page 3 – Speciality Insights
- Readmission by Medical Specialty (Shows which specialties have higher predicted readmission counts.)   
- Top Predictive Features (Displays which features most frequently appear as top predictors and their average SHAP impact values.)

### Page 4 – Clinical Indicators
- Readmission vs A1C Test Result (Highlights whether patients with higher A1C test results (poor glucose control) have higher predicted readmission risks.)
- Glucose Test vs Risk (Detects whether abnormal glucose levels correlate with higher predicted readmission risk.)

### Page 5 – Hospital Metrics
- Length of Stay vs Risk (Shows how readmission risk changes with longer hospital stays.)
- Procedures vs Risk (Shows whether patients who underwent more procedures have a higher predicted risk.)

---

## Requirements

### Python Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap joblib xgboost category_encoders
````

### Tools Used

* **Python 3.9+**
* **Jupyter Notebook / VS Code**
* **Power BI Desktop**
* **GitHub**

---

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/MohamedAdhil10/Hospital-Readmission-Prediction.git
   cd Hospital-Readmission-Prediction
   ```

2. Run the Python script:

   ```bash
   jupyter notebook hospital_readmission_analysis.ipynb
   ```

3. Open Power BI and import:

   * `dashboard_table.csv`

4. Recreate or explore visuals as described in the README.

---

## 📊 Model Performance

| Metric             | Score |
| :----------------- | :---: |
| ROC AUC (Tuned RF) | ~0.65 |
| PR AUC             | ~0.60 |
| Accuracy           | ~0.61 |
| Precision          | ~0.60 |
| Recall             | ~0.61 |

---

## Key Insights

* **A1C Test** and **Glucose Levels** are strong predictors of readmission.
* Patients with **longer hospital stays** tend to have higher readmission probabilities.
* Certain **medical specialties** show disproportionately high readmission rates.
* SHAP analysis provides transparent feature-level explanations for clinical decision support.

---

## Contact

* **Mohamed Adhil M**
* Email: [adhilm9991@gmail.com](mailto:adhilm9991@gmail.com)
* GitHub: [github.com/MohamedAdhil10](https://github.com/MohamedAdhil10)
* LinkedIn: [linkedin.com/in/mohamed-adhil-99118b247](https://linkedin.com/in/mohamed-adhil-99118b247)

---

*If you found this project helpful, please give it a star on GitHub!*
