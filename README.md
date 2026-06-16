# Multi-Module Data Mining Platform: Customer Churn Prediction & Market Basket Analysis

This interactive application combines Supervised Machine Learning for **Telecom Customer Churn Prediction** with an Unsupervised Data Mining engine for **Market Basket Analysis (MBA)**. The entire project is packaged into a user-friendly web interface.

---

## Interactive UI App (`app2.py`)
The project interface is built using **Streamlit**. Running `app2.py` opens a web dashboard with a clean sidebar to switch between the two modules:
* **Churn Prediction Portal:** Input customer info into simple forms to get instant churn risk predictions and see interactive model performance metrics.
* **Market Basket Analysis Dashboard:** Automatically reads your transaction data, lets you adjust threshold sliders, and instantly shows frequent item charts and rules.

---

## Module 1: Telecommunication Customer Churn Prediction

### Objective & Dataset
* **Goal:** Predict which telecom customers are likely to leave and provide retention insights.
* **Dataset:** IBM Telco Customer Churn (`Telco-Customer-Churn.csv`) featuring 7,043 customer records and 21 attributes.

### Data Preprocessing Pipeline
We tested models before and after applying these steps to see how preprocessing improves performance:
1.  **Imputation:** Replaced missing values in `TotalCharges` with the median.
2.  **Feature Drop:** Removed irrelevant columns (`customerID`).
3.  **Encoding & Scaling:** Handled categorical columns with `LabelEncoder` and normalized numeric values with `StandardScaler`.
4.  **Imbalance Fix:** Handled highly imbalanced data by applying **SMOTE** to boost model sensitivity.

### Model Evaluation (Primary Metric: Recall)
* **Decision Tree ($max\_depth=5$):** **Best Overall Model Selection**. Achieved a robust Recall of **0.8503** and the highest F1-Score of **0.8087** without overfitting.
* **Logistic Regression & SVM:** Showed the biggest jumps after preprocessing, scaling up their Recall scores by 46–49 points once features were balanced.

*For baseline experiments, see the local notebook:* `churn-prediction-.ipynb`

---

##  Module 2: Market Basket Analysis (MBA)

### Objective & Dataset
* **Goal:** Find hidden buying patterns and product associations in consumer purchases.
* **Dataset:** Groceries transactional database (`groceries.txt`) containing 9,835 transactions across 169 different items.

### Algorithm Comparison
Using a Minimum Support of `0.03` and a Minimum Confidence of `0.40`, we compared two popular mining engines:
* **Apriori vs. FP-Growth:** Both models generated the exact same **63 frequent itemsets** and **5 association rules**, proving mathematical equivalency.
* **Execution Time:** For this dataset size, **Apriori was faster**, finishing in **0.1497 seconds** compared to FP-Growth at **0.1977 seconds** (making Apriori roughly 1.32x faster here).
* **Top Discovered Rule:** `root vegetables ➔ other vegetables` (Lift = **2.2466**). This means a customer buying root vegetables is over twice as likely to buy other fresh vegetables.

---

## Key Business Recommendations
1.  **Targeted Retention:** Use machine learning probability scores to flag accounts with over a 60% risk factor for proactive customer service outreach.
2.  **Long-Term Strategy:** Move risky Month-to-Month contract users to stable annual contracts by offering short-term discounts.
3.  **Cross-Service Bundling:** Bundle high-value services together (like Tech Support and Online Security) to improve customer loyalty.
4.  **Smart Merchandising:** Arrange retail layouts or online recommendations to place high-lift products (like vegetables, milk, and yogurt) close together.

---

## Limitations & Future Upgrades
* Apply SMOTE strictly *after* splitting data to completely eliminate synthetic data leakage.
* Implement `GridSearchCV` to automatically tune model hyperparameters instead of using hardcoded depths.
* Upgrade to One-Hot Encoding for categorical features to prevent numerical relationships where they don't belong.

---

## Tools, Installation & Setup

### Requirements
Make sure you have Python 3.9+ installed. All essential libraries are listed inside your local `requirements.txt` file.

### Steps to Run
1. Navigate into your project folder.
2. Install all necessary dependencies:
   ```bash
   pip install -r requirements.txt
