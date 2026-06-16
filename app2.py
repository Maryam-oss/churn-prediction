"""
DSC306 – Data Mining  |  Terminal Project
Telecommunication Churn Prediction + Market Basket Analysis
Authors: Maryam Manahil (fa24-bds-053) | Ifrah Imran (fa24-bds-024)
COMSATS University Islamabad
"""

import time
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
)
from imblearn.over_sampling import SMOTE

# ── mlxtend ───────────────────────────────────────────────────────────────────

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DSC306 – Data Mining Project",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: train classifiers (cached so they are only trained once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training classifiers – please wait…")
def train_classifiers():
    """Load Telco data, preprocess, train 4 models, return everything needed."""
    try:
        df_raw = pd.read_csv("Telco-Customer-Churn.csv")
    except FileNotFoundError:
        return None  # handled in UI

    df = df_raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df.drop("customerID", axis=1, inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    dt  = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)
    svm = SVC(kernel="linear", random_state=42, probability=True).fit(X_train, y_train)
    lr  = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)

    def rule_based_scaled(row):
        if row["Contract"] < -0.8 and row["tenure"] < -0.5:
            return 1
        elif row["Contract"] < -0.8 and row["OnlineSecurity"] < 0:
            return 1
        elif row["Contract"] < -0.8 and row["MonthlyCharges"] > 0.5:
            return 1
        elif row["Contract"] > 0.5 and row["tenure"] > 0:
            return 0
        elif row["tenure"] > 1.0:
            return 0
        else:
            return 0

    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    y_pred_rb = X_test_df.apply(rule_based_scaled, axis=1).values

    def get_metrics(y_true, y_pred, name):
        return {
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

    metrics = pd.DataFrame([
        get_metrics(y_test, dt.predict(X_test),  "Decision Tree"),
        get_metrics(y_test, y_pred_rb,            "Rule-Based"),
        get_metrics(y_test, svm.predict(X_test),  "SVM"),
        get_metrics(y_test, lr.predict(X_test),   "Logistic Regression"),
    ]).set_index("Model")

    return {
        "dt": dt, "svm": svm, "lr": lr,
        "rule_based_fn": rule_based_scaled,
        "scaler": scaler,
        "feature_names": feature_names,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_df": X_test_df,
        "y_pred_rb": y_pred_rb,
        "df_raw": df_raw,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: load & encode groceries (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Groceries dataset…")
def load_groceries():

    # Read groceries.txt
    with open("groceries.txt", "r") as file:
        raw_data = file.read().splitlines()

    # Convert into transaction lists
    dataset = [row.split(",") for row in raw_data]

    # Transaction Encoding
    te = TransactionEncoder()
    te_array = te.fit(dataset).transform(dataset)

    # Convert to DataFrame
    df_groceries = pd.DataFrame(te_array, columns=te.columns_)

    return df_groceries


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAV
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("📊 DSC306 – Data Mining")
st.sidebar.markdown(
    "**Terminal Project**  \n"
    "Maryam Manahil · fa24-bds-053  \n"
    "Ifrah Imran · fa24-bds-024  \n"
    "*COMSATS University Islamabad*"
)
st.sidebar.markdown("---")
section = st.sidebar.radio(
    "Navigate to",
    ["Section A – Churn Classifier", "Section B – Market Basket Analysis"],
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A – CHURN CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
if section == "Section A – Churn Classifier":
    st.title("🔍 Section A – Customer Churn Prediction")
    st.markdown(
        "Enter customer attributes below, select a classifier, and click **Predict**."
    )

    result = train_classifiers()

    if result is None:
        st.error(
            "⚠️ `Telco-Customer-Churn.csv` not found. "
            "Place the CSV in the same folder as `app.py` and restart."
        )
        st.stop()

    df_raw = result["df_raw"]

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("predict_form"):
        st.subheader("Customer Attributes")
        c1, c2, c3 = st.columns(3)

        with c1:
            gender         = st.selectbox("Gender", ["Male", "Female"])
            senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner        = st.selectbox("Partner", ["Yes", "No"])
            dependents     = st.selectbox("Dependents", ["Yes", "No"])
            tenure         = st.slider("Tenure (months)", 0, 72, 12)
            phone_service  = st.selectbox("Phone Service", ["Yes", "No"])

        with c2:
            multiple_lines   = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security  = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
            online_backup    = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
            device_protect   = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
            tech_support     = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])

        with c3:
            streaming_tv      = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
            streaming_movies  = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
            contract          = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method    = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
            total_charges   = st.number_input("Total Charges ($)", 0.0, 9000.0, 800.0, step=10.0)

        st.markdown("---")
        classifier = st.selectbox(
            "Select Classifier",
            ["Decision Tree", "SVM", "Logistic Regression", "Rule-Based"],
        )
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        # Build a row matching the raw CSV structure (minus customerID & Churn)
        row = {
            "gender":           gender,
            "SeniorCitizen":    1 if senior == "Yes" else 0,
            "Partner":          partner,
            "Dependents":       dependents,
            "tenure":           tenure,
            "PhoneService":     phone_service,
            "MultipleLines":    multiple_lines,
            "InternetService":  internet_service,
            "OnlineSecurity":   online_security,
            "OnlineBackup":     online_backup,
            "DeviceProtection": device_protect,
            "TechSupport":      tech_support,
            "StreamingTV":      streaming_tv,
            "StreamingMovies":  streaming_movies,
            "Contract":         contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod":    payment_method,
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     total_charges,
        }

        # Encode using the same LabelEncoder fitted on the full dataset
        df_temp = df_raw.drop(columns=["customerID", "Churn"]).copy()
        df_temp["TotalCharges"] = pd.to_numeric(df_temp["TotalCharges"], errors="coerce")
        df_temp["TotalCharges"].fillna(df_temp["TotalCharges"].median(), inplace=True)

        le2 = LabelEncoder()
        encoded_row = {}
        for col in result["feature_names"]:
            if col == "SeniorCitizen":
                encoded_row[col] = row[col]
            elif df_temp[col].dtype == "object":
                le2.fit(df_temp[col])
                encoded_row[col] = le2.transform([row[col]])[0]
            else:
                encoded_row[col] = row[col]

        X_input = pd.DataFrame([encoded_row])[result["feature_names"]]
        X_scaled = result["scaler"].transform(X_input)

        # ── Predict ───────────────────────────────────────────────────────────
        if classifier == "Rule-Based":
            X_input_scaled_df = pd.DataFrame(X_scaled, columns=result["feature_names"])
            pred = result["rule_based_fn"](X_input_scaled_df.iloc[0])
            prob = None
        else:
            model_map = {
                "Decision Tree":       result["dt"],
                "SVM":                 result["svm"],
                "Logistic Regression": result["lr"],
            }
            model = model_map[classifier]
            pred  = model.predict(X_scaled)[0]
            prob  = model.predict_proba(X_scaled)[0][1] if hasattr(model, "predict_proba") else None

        label = "⚠️ CHURN" if pred == 1 else "✅ NO CHURN"
        color = "red" if pred == 1 else "green"

        st.markdown(f"### Prediction: :{color}[{label}]")

        if prob is not None:
            st.metric("Churn Probability", f"{round(prob*100, 1)}%")
        elif classifier == "Rule-Based":
            st.info("Rule-Based classifier uses deterministic rules – no probability output.")

        st.markdown("#### Model Evaluation Metrics (on test set)")
        styled = (
            result["metrics"].style
            .highlight_max(axis=0, props="background-color: #d4edda; color: #000000; font-weight: bold;")
        )
        st.dataframe(styled, use_container_width=True)

        # ── Classifier Comparison Bar Chart ───────────────────────────────────
        st.markdown("#### Classifier Comparison Chart")
        metrics_df = result["metrics"].reset_index()  # columns: Model, Accuracy, Precision, Recall, F1-Score
        metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
        models      = metrics_df["Model"].tolist()
        x           = np.arange(len(models))
        n_metrics   = len(metric_cols)
        bar_width   = 0.18
        palette     = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]  # one color per metric

        fig_cmp, ax_cmp = plt.subplots(figsize=(11, 5))
        for i, (metric, color) in enumerate(zip(metric_cols, palette)):
            vals   = metrics_df[metric].tolist()
            offset = (i - (n_metrics - 1) / 2) * bar_width
            bars   = ax_cmp.bar(x + offset, vals, width=bar_width,
                                label=metric, color=color, edgecolor="black", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax_cmp.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold", rotation=90
                )

        # Highlight the currently selected classifier's group
        selected_idx = models.index(classifier) if classifier in models else None
        if selected_idx is not None:
            ax_cmp.axvspan(
                selected_idx - bar_width * n_metrics / 2 - 0.05,
                selected_idx + bar_width * n_metrics / 2 + 0.05,
                color="yellow", alpha=0.18, zorder=0, label=f"Selected: {classifier}"
            )

        ax_cmp.set_xticks(x)
        ax_cmp.set_xticklabels(models, fontsize=11)
        ax_cmp.set_ylim(0, 1.08)
        ax_cmp.set_ylabel("Score", fontsize=11)
        ax_cmp.set_title("All Classifiers – Accuracy / Precision / Recall / F1-Score", fontsize=13)
        ax_cmp.legend(loc="lower right", fontsize=9)
        ax_cmp.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close()

        # ── Confusion matrix ──────────────────────────────────────────────────
        X_test    = result["X_test"]
        y_test    = result["y_test"]
        y_pred_rb = result["y_pred_rb"]

        if classifier == "Rule-Based":
            y_pr = y_pred_rb
        else:
            y_pr = model_map[classifier].predict(X_test)

        cm = confusion_matrix(y_test, y_pr)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=["Pred No", "Pred Yes"],
                    yticklabels=["Actual No", "Actual Yes"])
        ax_cm.set_title(f"Confusion Matrix – {classifier}")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close()

        # ── ROC Curve ─────────────────────────────────────────────────────────
        st.markdown("#### ROC Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))

        roc_models = {
            "Decision Tree":       result["dt"].predict_proba(X_test)[:, 1],
            "SVM":                 result["svm"].predict_proba(X_test)[:, 1],
            "Logistic Regression": result["lr"].predict_proba(X_test)[:, 1],
        }
        palette = {"Decision Tree": "steelblue", "SVM": "darkorange", "Logistic Regression": "green"}

        for name, proba in roc_models.items():
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)
            lw = 2.5 if name == classifier else 1.2
            ls = "-"  if name == classifier else "--"
            ax_roc.plot(fpr, tpr, color=palette[name], lw=lw, ls=ls,
                        label=f"{name} (AUC = {roc_auc:.3f})")

        # Rule-Based uses binary scores (no probability)
        fpr_rb, tpr_rb, _ = roc_curve(y_test, y_pred_rb)
        roc_auc_rb = auc(fpr_rb, tpr_rb)
        lw_rb = 2.5 if classifier == "Rule-Based" else 1.2
        ls_rb = "-"  if classifier == "Rule-Based" else "--"
        ax_roc.plot(fpr_rb, tpr_rb, color="purple", lw=lw_rb, ls=ls_rb,
                    label=f"Rule-Based (AUC = {roc_auc_rb:.3f})")

        ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curves – All Classifiers\n(selected model = solid line)")
        ax_roc.legend(loc="lower right", fontsize=8)
        ax_roc.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B – MARKET BASKET ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.title("🛒 Section B – Market Basket Analysis")
    st.markdown(
        "Groceries dataset (9,835 transactions, 169 unique items) loads automatically.  \n"
        "Adjust thresholds and click **Run Apriori** or **Run FP-Growth**."
    )

    df_groceries = load_groceries()
    st.success(f"Dataset loaded: {df_groceries.shape[0]:,} transactions × {df_groceries.shape[1]:,} items")

    # ── Controls ──────────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        min_support    = st.slider("Minimum Support",    0.01, 0.20, 0.03, 0.005)
    with col_r:
        min_confidence = st.slider("Minimum Confidence", 0.10, 0.90, 0.40, 0.05)

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    run_apriori   = btn_col1.button("▶ Run Apriori",         use_container_width=True)
    run_fpgrowth  = btn_col2.button("▶ Run FP-Growth",       use_container_width=True)
    run_both      = btn_col3.button("⚖️ Run Both & Compare", use_container_width=True)

    # ── Session state to persist results for comparison ───────────────────────
    if "apriori_result" not in st.session_state:
        st.session_state["apriori_result"] = None
    if "fpgrowth_result" not in st.session_state:
        st.session_state["fpgrowth_result"] = None

    def show_mba_results(freq_df, rules_df, runtime, algo_name, color):
        """Render bar chart, rules table, and summary for one algorithm."""
        top_rules = rules_df.sort_values("lift", ascending=False).head(10).reset_index(drop=True)

        # Summary panel
        m1, m2, m3 = st.columns(3)
        m1.metric("Frequent Itemsets", len(freq_df))
        m2.metric("Association Rules",  len(rules_df))
        m3.metric("Runtime (seconds)",  f"{runtime}s")

        # Bar chart
        st.subheader(f"{algo_name} – Top 15 Frequent Itemsets by Support")
        top15 = freq_df.nlargest(15, "support").copy()
        top15["label"] = top15["itemsets"].apply(lambda x: ", ".join(sorted(list(x))))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(top15["label"], top15["support"], color=color)
        ax.set_xlabel("Support")
        ax.set_title(f"{algo_name} – Top 15 Frequent Itemsets")
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Rules table
        if len(top_rules) == 0:
            st.warning("No rules generated at these thresholds. Try lowering the values.")
        else:
            st.subheader(f"Top-10 Association Rules by Lift")
            display_df = top_rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
            display_df["antecedents"]  = display_df["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
            display_df["consequents"]  = display_df["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
            display_df.columns         = ["Antecedent", "Consequent", "Support", "Confidence", "Lift"]
            display_df[["Support","Confidence","Lift"]] = display_df[["Support","Confidence","Lift"]].round(4)
            st.dataframe(display_df, use_container_width=True)

    def run_algorithm(algo, df_g, min_sup, min_conf):
        """Run apriori or fpgrowth, return (freq_df, rules_df, runtime) or (None, None, None)."""
        t0 = time.time()
        try:
            freq = apriori(df_g, min_support=min_sup, use_colnames=True) if algo == "apriori" \
                   else fpgrowth(df_g, min_support=min_sup, use_colnames=True)
            if len(freq) == 0:
                return None, None, None
            rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
            return freq, rules, round(time.time() - t0, 4)
        except Exception as e:
            st.error(f"Error: {e}")
            return None, None, None

    # ── Run Apriori ───────────────────────────────────────────────────────────
    if run_apriori:
        with st.spinner("Running Apriori…"):
            freq, rules, runtime = run_algorithm("apriori", df_groceries, min_support, min_confidence)
        if freq is None:
            st.warning("No frequent itemsets found. Lower the minimum support.")
        else:
            st.session_state["apriori_result"] = (freq, rules, runtime)
            st.success(f"Apriori completed in {runtime}s")
            show_mba_results(freq, rules, runtime, "Apriori", "steelblue")

    # ── Run FP-Growth ─────────────────────────────────────────────────────────
    if run_fpgrowth:
        with st.spinner("Running FP-Growth…"):
            freq, rules, runtime = run_algorithm("fpgrowth", df_groceries, min_support, min_confidence)
        if freq is None:
            st.warning("No frequent itemsets found. Lower the minimum support.")
        else:
            st.session_state["fpgrowth_result"] = (freq, rules, runtime)
            st.success(f"FP-Growth completed in {runtime}s")
            show_mba_results(freq, rules, runtime, "FP-Growth", "darkorange")

    # ── Run Both & Compare ────────────────────────────────────────────────────
    if run_both:
        with st.spinner("Running Apriori…"):
            freq_a, rules_a, time_a = run_algorithm("apriori",  df_groceries, min_support, min_confidence)
        with st.spinner("Running FP-Growth…"):
            freq_f, rules_f, time_f = run_algorithm("fpgrowth", df_groceries, min_support, min_confidence)

        if freq_a is None or freq_f is None:
            st.warning("No frequent itemsets found for one or both algorithms. Lower the minimum support.")
        else:
            st.session_state["apriori_result"]  = (freq_a, rules_a, time_a)
            st.session_state["fpgrowth_result"] = (freq_f, rules_f, time_f)
            st.subheader("Apriori Results")
            show_mba_results(freq_a, rules_a, time_a, "Apriori",   "steelblue")
            st.markdown("---")
            st.subheader("FP-Growth Results")
            show_mba_results(freq_f, rules_f, time_f, "FP-Growth", "darkorange")

    # ── Comparison chart (shown whenever both results are available) ──────────
    apr_res = st.session_state.get("apriori_result")
    fpg_res = st.session_state.get("fpgrowth_result")

    if apr_res is not None and fpg_res is not None:
        _, rules_apr, time_apr = apr_res
        _, rules_fpg, time_fpg = fpg_res

        st.markdown("---")
        st.subheader("📊 Apriori vs FP-Growth – Side-by-Side Comparison")

        algos    = ["Apriori", "FP-Growth"]
        runtimes = [time_apr, time_fpg]
        n_rules  = [len(rules_apr), len(rules_fpg)]
        colors   = ["steelblue", "darkorange"]

        fig_cmp, axes = plt.subplots(1, 2, figsize=(10, 4))

        bars0 = axes[0].bar(algos, runtimes, color=colors, edgecolor="black")
        axes[0].set_title("Runtime Comparison (seconds)", fontsize=12)
        axes[0].set_ylabel("Seconds")
        for bar, v in zip(bars0, runtimes):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         v + max(runtimes) * 0.02,
                         f"{v}s", ha="center", fontweight="bold")

        bars1 = axes[1].bar(algos, n_rules, color=colors, edgecolor="black")
        axes[1].set_title("Number of Rules Generated", fontsize=12)
        axes[1].set_ylabel("Rule Count")
        for bar, v in zip(bars1, n_rules):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         v + max(max(n_rules), 1) * 0.02,
                         str(v), ha="center", fontweight="bold")

        fig_cmp.suptitle(
            f"Apriori vs FP-Growth  |  Support = {min_support}  Confidence = {min_confidence}",
            fontsize=13
        )
        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close()

        faster  = "Apriori" if time_apr < time_fpg else "FP-Growth"
        faster_t = min(time_apr, time_fpg)
        speedup  = round(max(time_apr, time_fpg) / faster_t, 2) if faster_t > 0 else "N/A"
        c1, c2 = st.columns(2)
        c1.info(f"**Faster algorithm:** {faster}  \nSpeed-up: **{speedup}×**")
        c2.info(f"Apriori → {len(rules_apr)} rules in {time_apr}s  \nFP-Growth → {len(rules_fpg)} rules in {time_fpg}s")

    elif apr_res is not None or fpg_res is not None:
        st.markdown("---")
        st.info("💡 Run **both** algorithms (or click **Run Both & Compare**) to see the side-by-side comparison chart.")
