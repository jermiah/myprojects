# Length of Stay (LOS) Prediction Project - Under development

Codes are available in [LOS Prediction Notebook](./Code)

## **Problem Statement**
The recent COVID-19 pandemic has highlighted the importance of effective healthcare management. One critical aspect of healthcare management is the **Length of Stay (LOS)** of patients. Accurately predicting LOS can significantly improve hospital efficiency by aiding in resource allocation, room and bed planning, and minimizing risks such as staff and visitor infection.

This project focuses on predicting the LOS for each patient at the time of admission to optimize treatment plans and streamline hospital operations. The goal is to classify LOS into one of 11 categories, ranging from **0–10 days** to **more than 100 days**.

You are tasked with solving this problem as a **Data Scientist** for **HealthMan**, a non-profit organization dedicated to professional hospital management.

---

## **Data Description**
The project uses the following datasets:

- **`train_data.csv`**: Contains features related to patient and hospital data, along with the LOS classification.
- **`train_data_dictionary.csv`**: Provides metadata and detailed descriptions of the features in `train_data.csv`.
- **`test_data.csv`**: Includes patient and hospital features where LOS needs to be predicted.

### **Target Variable**
The LOS target variable is divided into the following 11 classes:
1. 0–10 days
2. 11–20 days
3. 21–30 days
4. 31–40 days
5. 41–50 days
6. 51–60 days
7. 61–70 days
8. 71–80 days
9. 81–90 days
10. 91–100 days
11. More than 100 days

## **Project Structure**
The project involves the following steps:

1. **Exploratory Data Analysis (EDA)**:
   - Understand the dataset and its features.
   - Visualize key trends and patterns in LOS.

2. **Data Preprocessing**:
   - Handle missing values.
   - Encode categorical variables.
   - Scale numerical features as necessary.

3. **Model Development**:
   - Train multiple classification models such as RandomForest, GradientBoosting, XGBoost, and CatBoost.
   - Evaluate models using accuracy, precision, recall, and F1-score.

4. **Prediction and Submission**:
   - Generate LOS predictions for test cases.
   - Create a submission file in the required format.

---

## **Key Inferences from Model Evaluation**
### **Model Evaluation Results**
The following insights were derived from training and evaluating models:

- **CatBoost**:
  - Achieved the best ROC AUC during hyperparameter tuning.
  - Model performance on the test set:
    - Accuracy
    - Precision
    - Recall
    - F1 Score

- **LightGBM**:
  - Achieved the second-best ROC AUC.
  - Model performance on the test set:
    - Accuracy
    - Precision
    - Recall
    - F1 Score

- **XGBoost**:
  - Delivered comparable results with an ROC AUC.
  - Model performance on the test set:
    - Accuracy
    - Precision
    - Recall
    - F1 Score

These results highlight the effectiveness of CatBoost, LightGBM, and XGBoost for this multiclass classification problem, with CatBoost being the most promising model.

---

## **Acknowledgements**
This problem statement and dataset were sourced from a hackathon hosted on **Analytics Vidhya**. You can find more details about the challenge [here](https://datahack.analyticsvidhya.com/contest/janatahack-healthcare-analytics-ii/#ProblemStatement).

---


---

## **Purpose**
This project demonstrates the application of **machine learning in healthcare analytics**, showcasing how data-driven decisions can improve operational efficiency and patient care in hospitals.
