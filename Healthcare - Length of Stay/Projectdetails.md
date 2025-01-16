# Length of Stay (LOS) Prediction Project

Codes are available in [LOS Prediction Notebook](./Code)

## **Introduction**
This analysis focuses on predicting Patients' Length of Stay (LOS) during the COVID-19 Pandemic. The COVID-19 pandemic overloaded hospital care resources and efficient resource management and allocation are vital to stablizing the healthcare system. The objective of this analysis is to develop machine learning models to predict LOS to help improve resource allocation since LOS is an important indiactor for monitoring health management processes.

Dataset: The dataset comes from Kaggle. A version of the dataset was originally used for Analytics Vidhya Healthcare Analytics Hackathon. The target variable for the dataset is Stay Days which is categorical with 9 categories representing range of days (length of stay in the hospital) between 0 to 100+ days.

This analysis is an extension as well as a response to an analysis performed by The Jianing Pei et al. which uses the Analytics Vidha Healthcare Analytics Hackathon dataset to predict patient's length of stay. In their analysis, their models achieved an accuracy score of 0.3541 using an optimized Random Forest algorithm. 

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

## **Acknowledgements**
This problem statement and dataset were sourced from a hackathon hosted on **Analytics Vidhya**. You can find more details about the challenge [here](https://datahack.analyticsvidhya.com/contest/janatahack-healthcare-analytics-ii/#ProblemStatement).


## **Purpose**
This project demonstrates the application of **machine learning in healthcare analytics**, showcasing how data-driven decisions can improve operational efficiency and patient care in hospitals.
