# Internship Submission: Student Performance Tracker using Machine Learning
**Author:** Machine Learning Engineering Intern  
**Date:** June 2026  

---

## 1. Executive Summary

Academic performance prediction is a critical task in educational data mining. It enables educational institutions to implement early interventions, allocate tutoring resources, and support students before final examinations occur. 

This project develops an intermediate-level machine learning regression system to predict a student's **Final Exam Score (0 - 100%)** based on various demographic, behavioral, and academic features. By utilizing the real **UCI Student Performance Dataset**, mapping variables to clean targets, training multiple regression models, and embedding the selected model in a **Streamlit** dashboard, this system provides school administrators and parents with actionable predictions and custom study recommendations.

---

## 2. Dataset Description & Prep

The system leverages the math course student records from the UCI Student Performance repository (`student-mat.csv`). The dataset contains 395 student instances and 33 attributes, reflecting family environment, school circumstances, and periodic grades.

For this submission, the raw data was transformed into a cleaned schema (`student_performance.csv`) containing the following 10 features and 1 target variable:

### 2.1 Feature Schema
| Feature | Type | Source / Derivation | Description |
| :--- | :--- | :--- | :--- |
| **Gender** | Categorical | `sex` | 'Male' or 'Female' |
| **Age** | Numerical | `age` | Student age (15 to 22) |
| **Study_Hours** | Numerical | `studytime` | Categorical 1-4 mapped to continuous weekly study hours |
| **Attendance** | Numerical | `absences` | Derived percentage: $100.0 - (\text{absences} \times 1.5)$ capped at 0% |
| **Previous_Score** | Numerical | `G1` | First-period exam score scaled to 0-100 (multiplied by 5) |
| **Parent_Education** | Categorical | `max(Medu, Fedu)` | Maximum education of either parent (mapped to levels) |
| **Internet_Access** | Categorical | `internet` | Home internet access ('Yes' or 'No') |
| **Extra_Activities** | Categorical | `activities` | Engagement in school extra-curriculars ('Yes' or 'No') |
| **Family_Support** | Categorical | `famsup` | Educational support from family ('Yes' or 'No') |
| **Sleep_Hours** | Numerical | Generated | Modeled using freetime, studytime, goout, and random noise |
| **Final_Score** | Target (Numeric) | `G3` | Final math score scaled to 0-100 (multiplied by 5) |

### 2.2 Imputation & Cleaning Simulation
To demonstrate standard data cleaning:
- **Duplicates:** 5 random duplicate rows were injected into the raw file.
- **Missing Values:** ~2% random nulls (NaNs) were injected in `Attendance`, `Study_Hours`, `Sleep_Hours`, and `Parent_Education` columns.

The `src/preprocessing.py` script automatically scans for and drops duplicate rows, then imputes missing numeric values with column medians and categorical values with column modes.

---

## 3. Exploratory Data Analysis (EDA) Insights

Exploratory Data Analysis was performed, generating 6 key visualizations stored under `notebooks/plots/`:

1. **Correlation Heatmap:** 
   - Displays linear correlations among numerical features. 
   - **Key Finding:** *Previous Score* ($r \approx 0.80$) and *Attendance* ($r \approx 0.22$) are the primary positive linear correlates with *Final Score*.
2. **Score Distribution:** 
   - Shows exam scores grouped around 50-65% with a secondary mode at 0% (representing students who failed to attend the final exam).
3. **Attendance & Study Hours Analysis:**
   - Visualizes positive skewness in attendance (most students maintain >90% attendance).
   - Scatter plots of weekly study hours against final scores demonstrate that students with higher study hours consistently score higher.
4. **Boxplot Comparisons:**
   - Boxplots segmenting final scores by categorical factors indicate that higher *Parent Education* levels and the presence of *Family Support* are associated with increased median student scores.

---

## 4. Data Preprocessing Pipeline

Data preprocessing follows a strict train-test isolation protocol. Standard parameters are fit on the training split ($80\%$) and applied to the test split ($20\%$) to prevent data leakage.

```
Raw Data -> Drop Duplicates -> Impute NaNs (Median/Mode) -> Custom Encoding -> Numeric Scaling -> Train-Test Split
```

### 4.1 Custom Categorical Mappings
Rather than using generic label encoding, custom mappings were designed to preserve the physical structure and order of variables:
- **Gender:** `{'Male': 0, 'Female': 1}`
- **Internet_Access, Extra_Activities, Family_Support:** `{'No': 0, 'Yes': 1}`
- **Parent_Education (Ordinal Mapping):**
  `{'None': 0, 'Primary Education': 1, '5th to 9th Grade': 2, 'Secondary Education': 3, 'Higher Education': 4}`

### 4.2 Feature Scaling
Numerical columns are scaled using standard normalization (z-score scaling):
\[
z = \frac{x - \mu}{\sigma}
\]
where $\mu$ is the training mean and $\sigma$ is the training standard deviation. Features scaled include `Age`, `Study_Hours`, `Attendance`, `Previous_Score`, and `Sleep_Hours`.

---

## 5. Model Training & Comparison

Five diverse regression algorithms from `scikit-learn` were trained on the preprocessed training set ($N=316$) and evaluated on the test set ($N=80$):

1. **Linear Regression:** Baseline parametric estimator.
2. **Decision Tree Regressor:** Non-parametric partitioner (hyperparameter `max_depth=5` to prevent overfitting).
3. **Random Forest Regressor:** Bootstrap ensemble of 100 decision trees (`max_depth=6` to regulate complexity).
4. **Support Vector Regressor (SVR):** Distance-based kernel regressor (RBF kernel, $C=10.0$, $\epsilon=0.1$).
5. **Gradient Boosting Regressor:** Sequential boosting estimator (100 boosting stages, learning rate $=0.08$, `max_depth=4`).

### 5.1 Performance Comparison Table
*Evaluation metrics computed on test set:*

| Rank | Model | $R^2$ Score | MAE | MSE | RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | **Random Forest Regressor** | **0.7192** | **8.9182** | **144.4194** | **12.0175** |
| 2 | Gradient Boosting Regressor | 0.7069 | 9.0157 | 150.7447 | 12.2778 |
| 3 | Linear Regression | 0.6759 | 9.6244 | 166.7080 | 12.9115 |
| 4 | Support Vector Regressor | 0.6380 | 9.5741 | 186.1962 | 13.6454 |
| 5 | Decision Tree Regressor | 0.5906 | 9.7038 | 210.5715 | 14.5111 |

---

## 6. Evaluation of the Selected Model (Random Forest)

The system automatically selected the **Random Forest Regressor** as the best model due to its superior coefficient of determination ($R^2 \approx 0.719$).

### 6.1 Visualization Analysis
The evaluation script (`src/evaluate.py`) exported three key evaluation graphs to `notebooks/plots/`:

- **Actual vs. Predicted Scores:** Scatter points are distributed tightly along the 45-degree identity line. The model tracks trends well across the entire grade spectrum, though it exhibits slight regression-to-the-mean for extremely low and high outliers.
- **Residuals Plot:** Shows random scatter centered around zero, indicating that residuals are homoscedastic (constant variance) and that the model is free from systematic bias.
- **Feature Importance:**
  - **Previous Score** represents the most dominant indicator (relative importance $> 65\%$).
  - **Attendance** and **Study Hours** represent the secondary and tertiary most important indicators.
  - Socio-demographic features (Gender, Family Support) show lower direct influence, which makes physical sense as they act as indirect drivers.

---

## 7. Interactive Streamlit Dashboard

The Streamlit web application (`app.py`) serves as the user-facing deployment. Its architecture is divided into three key panels:

1. **🔮 Score Predictor:**
   - Interactive forms and sliders allow users to input new student profiles.
   - Computes predictions and renders cards displaying the **Predicted Final Score** and **Performance Category**:
     - *Excellent* ($\ge 85\%$)
     - *Good* ($70\% - 84.9\%$)
     - *Average* ($55\% - 69.9\%$)
     - *Needs Improvement* ($< 55\%$)
   - Calculates a **95% Confidence Interval** ($\pm 1.96 \times RMSE \approx \pm 23.6\%$) using the model's test RMSE as the standard error, highlighting statistical rigor.
   - Generates contextual, dynamic study recommendations. For instance, low attendance or short study hours trigger specific warnings and remediation steps.
2. **📊 Dataset Insights:**
   - Renders live summaries of the underlying student dataset.
   - Contains interactive distribution plot selectors.
3. **⚙️ Model Performance:**
   - Renders the model leaderboard and imports the saved evaluation plots directly into the interface.

---

## 8. Conclusion

This project successfully implements a complete, end-to-end Machine Learning system for predicting student performance. The system demonstrates key data science engineering methodologies:
- Clean dataset preparation based on real-world education records.
- Robust preprocessing addressing duplicates, missing values, custom encodings, and scaling.
- Comprehensive model validation leading to the automatic selection of a Random Forest regressor.
- Production-ready web deployment with actionable feedback.

### Future Recommendations
- **Classification Modeling:** Extend the training script to include classification (Pass/Fail) predictions.
- **Feature Expansion:** Collect additional temporal factors, such as daily screen time and mental health indicators.
- **Deep Learning:** Investigate multi-layer perceptron (MLP) architectures for larger student databases.
