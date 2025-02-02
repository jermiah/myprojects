# Data Science and AI Projects

## **Overview**
This repository contains a collection of Machine Learning (ML), Deep Learning (DL), and Optimization projects, covering various domains and real-world applications. Each project explores different models, techniques, and libraries to solve complex problems efficiently.

---

For Git and GitHub setup:
- [GitHub Repository Setup](./Docs/Github.md)

## **Project Structure**

```plaintext
/
├── README.md                       # Main documentation
├── docs/
│   ├── Github.md                    # Git and GitHub setup instructions
├── Bankruptcy_Prediction            # A project on predicting bankruptcy for companies
├── Healthcare - Length of Stay      # A project on predicting the length of stay of a patient
├── Cool Wipes - Linear Programming  # A new project related to Cool Wipes
└── ...
```

## **Key Features**

- **Diverse Domains:** Healthcare, finance, e-commerce, and more.
- **Model Categories:**
  - Regression and Classification Models
  - Clustering and Dimensionality Reduction
  - Time Series Models (ARIMA, SARIMA, Prophet, etc.)
  - Deep Learning (CNNs, RNNs, Transformers, GANs)
  - Ensemble Models (Random Forest, XGBoost, LightGBM)
- **Libraries and Tools:** pandas, NumPy, Matplotlib, Seaborn, scikit-learn, TensorFlow, PyTorch, Keras, statsmodels, and more.
- **Complete Workflows:**
  - Problem definition
  - Data preprocessing
  - Feature engineering
  - Model development and evaluation
  - Insights and interpretation

---
### **1. Bankruptcy Prediction**

#### Summary
Predicts the likelihood of bankruptcy for companies to aid in risk assessment and financial planning.

#### Highlights
- **Data:** Financial and operational metrics of companies with 96 variables and 6819 records. The target variable indicates bankruptcy status.
- **Methods:** Logistic Regression, XGBoost, and Ensemble Methods with hyperparameter tuning for improved accuracy.
- **Evaluation:** Accuracy, Precision, Recall, F1 Score, and model interpretation using ensemble predictions.

---
**Project Directory:**  [Bankruptcy Prediction](./Bankruptcy%20Prediction/Projectdetails.md)

---

### **2. Patient Length of Stay Prediction**

#### **Summary:**
Predicts hospital patient length of stay (LOS) to enhance resource allocation and care management.

#### **Highlights:**
- **Data:** Patient and hospital records categorized into 11 LOS classes.
- **Methods:** Logistic Regression, Random Forest, Gradient Boosting, LightGBM, CatBoost, SVM, XGBoost.
- **Evaluation:** Accuracy, Precision, Recall, F1 Score, ROC AUC.
---
**Project Directory:**  [Patient Length of Stay Prediction](./Healthcare%20-%20Length%20of%20Stay/Projectdetails.md)

---

### **3. Cool Wipes - Linear Programming with Gurobi**

#### **Summary**
Optimizing the production and distribution network for CoolWipes using **Gurobi** to minimize costs while meeting demand across six geographic regions.

#### **Highlights**
- **Data:** Demand data for wipes and ointments across six regions, current plant capacities, fixed and variable costs, and transportation costs.
- **Methods:** 
  - **Gurobi Optimization:** Formulating a linear programming (LP) model to determine the optimal production and distribution strategy.
  - **Scenario Analysis:** Evaluating cost structures under different transportation cost assumptions.
- **Evaluation:**
  - **Baseline Analysis:** Assessing the annual cost of serving the entire nation from Chicago.
  - **Expansion Decision:** Evaluating the impact of adding new plants in Princeton, Atlanta, or Los Angeles.
  - **Optimal Network Design:** Recommending the best plant locations and capacities under varying constraints.
  - **Future Planning:** Projecting network structure for 2026 with a 35% demand increase and next-day delivery.

#### **Key Questions Addressed**
- What is the cost of serving the nation from a single plant in Chicago?
- Should additional plants be built? If so, where and with what capacity?
- How does transportation cost variability influence plant location decisions?
- What is the best network design if starting from scratch?
- Can the network support projected demand growth by 2026?
- How can AI and smart technologies improve supply chain efficiency?

---
**Project Directory:**  [Cool Wipes - Linear Programming](./Cool%20Wipes%20-%20Linear%20Programming)

---

## **Contributing**
Contributions are welcome! Fork the repository, make your changes, and submit a pull request. For major changes, open an issue for discussion.

---
