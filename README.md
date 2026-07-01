# Student Performance Tracker using Machine Learning

## Intern Details

- **Intern ID:** CMPTBD7AX0
- **Full Name:** Divyansh Rai
- **No. of Weeks:** 1 Week
- **Project Name:** Student Performance Tracker
- **Project Scope:** This project predicts students' final academic performance using machine learning regression algorithms. It includes data preprocessing, exploratory data analysis, model training, evaluation, and a Streamlit web application for real-time score prediction.


This is an intermediate-level machine learning project designed to predict student academic performance based on socio-demographic, behavioral, and academic features. It uses the real **UCI Student Performance Dataset** (Math course) and maps the attributes to student performance indicators. 

The project features a modular code structure, automated model training and selection, comprehensive evaluation plots, and an interactive **Streamlit** dashboard.

---

## 📂 Project Structure

```
student/
│
├── dataset/
│   ├── prepare_dataset.py
│   ├── student-mat.csv
│   ├── student_performance.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── models/
│   ├── best_model.joblib
│   ├── best_model_metadata.joblib
│   ├── scaler.joblib
│   ├── imputation_values.joblib
│   ├── model_comparison.csv
│   ├── linear_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   └── svr.joblib
│
├── notebooks/
│   ├── EDA.ipynb
│   └── plots/
│       ├── actual_vs_predicted.png
│       ├── attendance_distribution.png
│       ├── boxplots_analysis.png
│       ├── correlation_heatmap.png
│       ├── feature_importance.png
│       ├── histograms_numerical.png
│       ├── residuals_plot.png
│       ├── score_distribution.png
│       └── study_hours_vs_score.png
│
├── screenshots/
│   ├── dashboard.png
│   ├── input_form.png
│   └── prediction_result.png
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── README.md
└── report.md
```

---

## 🛠️ Tech Stack & Requirements

- **Language:** Python 3.13+
- **Libraries:**
  - `pandas` (Data loading and manipulation)
  - `numpy` (Numerical operations)
  - `matplotlib` & `seaborn` (Statistical visualization)
  - `scikit-learn` (Imputation, Scaling, Model Training, Cross-Validation)
  - `joblib` (Model persistence)
  - `streamlit` (Dashboard application UI)

To install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution & Run Guide

Follow these steps sequentially to generate the dataset, train models, evaluate, and run the Streamlit app.

### Step 1: Download & Prepare the Dataset
Downloads the raw UCI dataset, maps variables, injects a small set of duplicates & missing values, and saves the file:
```bash
python dataset/prepare_dataset.py
```

### Step 2: Train the Regressors
Cleans, imputes missing values, encodes categories, splits data 80:20, trains 5 regressors, and selects the best performer:
```bash
python src/train.py
```

### Step 3: Run the Model Evaluation
Evaluates the selected model, computes metrics ($R^2$, MAE, MSE, RMSE), and generates residual, actual vs. predicted, and feature importance plots:
```bash
python src/evaluate.py
```

### Step 4: Run CLI Prediction (Verification)
Verify the prediction pipeline works on a mock student profile:
```bash
python src/predict.py --sample
```

### Step 5: Start the Streamlit Web Application
Launches the interactive user interface in your default web browser:
```bash
streamlit run app.py
```

---

## ⚙️ Model Comparison Summary

Five regressors were trained and compared. The summary results are sorted by $R^2$ descending:

| Rank | Model | $R^2$ Score | MAE | MSE | RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | **Random Forest Regressor** | **0.7192** | **8.9182** | **144.4194** | **12.0175** |
| 2 | Gradient Boosting Regressor | 0.7069 | 9.0157 | 150.7447 | 12.2778 |
| 3 | Linear Regression | 0.6759 | 9.6244 | 166.7080 | 12.9115 |
| 4 | Support Vector Regressor (SVR) | 0.6380 | 9.5741 | 186.1962 | 13.6454 |
| 5 | Decision Tree Regressor | 0.5906 | 9.7038 | 210.5715 | 14.5111 |

*Note: Individual runs might show slight differences due to random sampling, but Random Forest consistently scores the highest accuracy ($R^2 \approx 72\%$).*

---

## 💡 Key Highlights

1. **Robust Custom Preprocessing:** Instead of arbitrary numbers, ordinal mapping was used for education levels and binary encoding for boolean values, preserving structural correlations.
2. **Missing & Duplicate Auditing:** The preprocessing module handles duplicate instances and imputes missing values (using median/mode statistics), representing real-world production engineering.
3. **Streamlit UI Integration:** Incorporates metric cards, interactive plots, side-by-side configurations, and contextual, personalized recommendations based on student weaknesses.
4. **Prediction Bounds:** Predictions output a 95% Confidence Interval ($Score \pm 1.96 \times RMSE$) to represent uncertainty, reinforcing machine learning rigor.
