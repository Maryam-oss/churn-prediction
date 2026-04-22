# Telecommunication Customer Churn Prediction

This project focuses on predicting customer churn in the telecommunications industry using machine learning techniques. The goal is to identify customers who are likely to leave the company and provide actionable insights for improving customer retention.

---

## Objective

* Predict customer churn using classification models
* Analyze the impact of data preprocessing on model performance
* Compare multiple machine learning models
* Provide business insights to reduce churn

---

## Dataset

* **Dataset:** IBM Telco Customer Churn
* **Source:** Kaggle
* **Total Records:** 7,043 customers
* **Features:** 21 (including target variable `Churn`)

The dataset contains customer information such as:

* Demographics (gender, senior citizen)
* Account details (tenure, contract type)
* Services (internet, security, streaming)
* Billing information (monthly & total charges)

---

## Data Preprocessing

The following preprocessing steps were applied:

* Handled missing values in `TotalCharges` using median imputation
* Removed irrelevant feature (`customerID`)
* Encoded categorical variables using Label Encoding
* Applied feature scaling using StandardScaler
* Handled class imbalance using SMOTE

These steps significantly improved model performance, especially Recall.

---

## Models Implemented

The following models were trained and evaluated:

* Decision Tree
* Rule-Based Classifier
* Support Vector Machine (SVM)
* Logistic Regression

Each model was evaluated **before and after preprocessing** to analyze performance improvements.

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall (Most Important)
* F1-Score

Recall is prioritized because missing a churner is more costly than false alarms.

---

## Results Summary

After applying preprocessing:

* All models showed significant improvement
* SVM and Logistic Regression improved the most
* Logistic Regression achieved the best overall balance

---

## Best Model: Logistic Regression

Logistic Regression was selected as the best model because:

* High Recall and F1-score
* Provides probability of churn (useful for business decisions)
* Interpretable coefficients
* Strong performance after preprocessing

---

## Key Business Insights

* Month-to-month contract customers are most likely to churn
* New customers (low tenure) are high-risk
* Lack of security and tech support increases churn
* High monthly charges increase churn probability

---

## Recommendations

* Offer discounts to convert users to long-term contracts
* Improve onboarding for new customers
* Bundle services like security & tech support
* Target high-risk customers using churn probability

---

## Limitations

* Label encoding may introduce unintended relationships
* SMOTE applied before train-test split
* No hyperparameter tuning
* Single train-test split used

---

## Future Improvements

* Use One-Hot Encoding
* Apply cross-validation
* Perform hyperparameter tuning
* Try ensemble models (Random Forest, XGBoost)

---

## Tools & Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Matplotlib, Seaborn

---

## References

* IBM Telco Dataset (Kaggle)
* Scikit-learn Documentation
* SMOTE Research Paper

---


