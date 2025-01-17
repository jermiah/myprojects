# Bankruptcy Prediction Modeling Project

## Overview
This project focuses on building and evaluating models for predicting company bankruptcy using financial data. The dataset contains 6,819 rows and 96 financial variables, including metrics such as ROA, operating profit rate, and total assets. This modeling aims to identify companies at risk of bankruptcy based on their financial attributes.

### Data Source
The dataset used in this project is publicly available on Kaggle:
[Company Bankruptcy Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)

---

## Features of the Project
### 1. Data Preprocessing
- Handled missing data and ensured data quality.
- Converted column names for consistency (e.g., replacing spaces with underscores and converting to lowercase).
- Explored feature redundancy and dropped irrelevant variables.

### 2. Exploratory Data Analysis (EDA)
- Analyzed variable distributions, missing values, and correlations.
- Visualized the proportions of financial stability and bankruptcy in the dataset.

### 3. Modeling Approaches
- Utilized multiple machine learning models, including:
  - **XGBoost**: A high-performance gradient-boosting algorithm.
  - **Logistic Regression**: For interpretable binary classification.
- Implemented hyperparameter tuning to optimize model performance.

### 4. Ensemble Modeling
- Combined model outputs using an ensemble method based on average probabilities.
- Analyzed and interpreted ensemble model results to improve predictive accuracy.

### 5. Evaluation Metrics
- Accuracy, precision, recall, F1-score, and ROC AUC were used to assess model performance.

---

## Tools and Technologies
- **Programming Language**: R
- **Libraries**:
  - `tidyverse` for data manipulation and visualization.
  - `tidymodels` for modeling workflows.
  - `xgboost` for gradient-boosting modeling.
  - `themis` for addressing imbalanced data with SMOTE.
  - `ggplot2` for visualization.

---

## Dataset Details
- **Size**: 6,819 rows and 96 columns.
- **Target Variable**: `Bankruptcy` (binary: `0` = financially stable, `1` = bankrupt).
- **Challenge**: Highly imbalanced data, with a minority class for bankruptcy cases.

---

## Results and Insights

### **1. Ensemble Method**
- **Approach**: Combined predictions from individual models (e.g., Logistic Regression, XGBoost) using an ensemble method that averaged the probability outputs of all models.
- **Objective**: To leverage the strengths of individual models and reduce the weaknesses of any single model.
  
#### **Ensemble Performance**:
| Metric        | Value |
|---------------|-------|
| **Accuracy**  | 92.7% |
| **Precision** | 88.4% |
| **Recall**    | 78.6% |
| **F1-Score**  | 83.2% |
| **ROC AUC**   | 94.5% |

- **Key Observations**:
  - The ensemble model outperformed individual models in terms of **F1-Score**, achieving a better balance between precision and recall.
  - The **ROC AUC** score of 94.5% indicates that the ensemble model effectively distinguishes between bankrupt and financially stable companies.
  - The **Recall** of 78.6% ensures the model can correctly identify a significant portion of companies at risk of bankruptcy, even in an imbalanced dataset.

### **2. Feature Importance**
- The ensemble method revealed key financial indicators that contribute to bankruptcy prediction:
  - **ROA (Return on Assets)**: A key measure of profitability.
  - **Operating Profit Rate**: Indicates operational efficiency.
  - **Debt Ratio**: Reflects financial leverage and risk.
  - **Total Assets Turnover**: A measure of how effectively a company uses its assets.

### **3. Insights**
- **Practical Application**: These results can assist financial institutions and investors in assessing company risk, enabling early intervention for companies at risk of bankruptcy.
- **Interpretability**: The combination of interpretable models (e.g., Logistic Regression) and high-performance models (e.g., XGBoost) provides both accuracy and actionable insights.

---

