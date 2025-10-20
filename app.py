# =========================================================
# Credit Card Fraud Detection Dashboard
# Supervised (LR/RF + LIME + SMOTE) + Unsupervised (Isolation Forest)
# =========================================================

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
import lime
import lime.lime_tabular

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("---")

st.markdown("""
### ℹ️ About This Dashboard
This dashboard demonstrates **two fraud detection strategies**:
- **Supervised learning** (when your dataset has a target column `IsFraud`)
- **Unsupervised learning** (when the dataset has no target, only numeric features)

👉 Upload your CSV or test with the default Kaggle or Synthetic dataset.
""")

# -----------------------------
# Sidebar: Dataset Options
# -----------------------------
st.sidebar.header("⚙️ Settings")

# Dataset selector
dataset_choice = st.sidebar.selectbox(
    "Select Default Dataset:",
    ["Kaggle Dataset (99% legit)", "Synthetic Dataset (5% fraud)"]
)

uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type=["csv"])

if uploaded_file is not None:
    st.sidebar.success("✅ File uploaded successfully!")
    data = pd.read_csv(uploaded_file)
elif dataset_choice == "Synthetic Dataset (5% fraud)":
    data = pd.read_csv("data/synthetic_fraudulent_credit_card_transactions.csv")
else:
    data = pd.read_csv("data/credit_fraud_dataset.csv")

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("📂 Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

# -----------------------------
# Helper: Pie Chart
# -----------------------------
def plot_pie(series, labels=("Legit", "Fraud"), title="Fraud vs Legit"):
    counts = series.value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct="%1.1f%%",
           colors=["#4CAF50", "#FF5252"], startangle=90)
    ax.set_title(title)
    st.pyplot(fig)

# -----------------------------
# Auto Mode Selection
# -----------------------------
has_target = "IsFraud" in data.columns

mode_choice = st.sidebar.radio(
    "Learning Mode",
    ("Auto", "Supervised (requires IsFraud)", "Unsupervised (Isolation Forest)")
)

def resolve_mode(choice, has_target_flag):
    if choice == "Auto":
        return "supervised" if has_target_flag else "unsupervised"
    if choice.startswith("Supervised") and not has_target_flag:
        st.sidebar.warning("⚠️ No IsFraud column found. Switching to Unsupervised mode.")
        return "unsupervised"
    return "supervised" if choice.startswith("Supervised") else "unsupervised"

mode = resolve_mode(mode_choice, has_target)

# -----------------------------
# UNSUPERVISED: Isolation Forest
# -----------------------------
def run_unsupervised(df):
    st.warning("🧠 Running Unsupervised Mode (Isolation Forest — no `IsFraud` column found).")

    # Use numeric features
    X = df.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        st.error("⚠️ No numeric columns found for anomaly detection.")
        st.stop()

    # Sidebar control
    st.sidebar.subheader("🧪 Isolation Forest Settings")
    contamination = st.sidebar.slider(
        "Assumed Fraud Rate (contamination)", min_value=0.01, max_value=0.20,
        value=0.05, step=0.01, help="Estimated proportion of anomalies/fraud."
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
    preds = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)

    df["AnomalyScore"] = scores
    df["IsFraud_Predicted"] = pd.Series(preds).map({1: 0, -1: 1})

    st.subheader("📊 Detected Fraud Distribution (Unsupervised)")
    plot_pie(df["IsFraud_Predicted"], title="Predicted Fraud vs Legit (Isolation Forest)")

    st.subheader("🔍 Top Suspicious Transactions")
    top = df[df["IsFraud_Predicted"] == 1].sort_values("AnomalyScore").head(10)
    st.dataframe(top, use_container_width=True)

    st.info("Isolation Forest identifies **outliers** as potential frauds based on pattern deviations.")

# -----------------------------
# SUPERVISED: Logistic Regression / Random Forest
# -----------------------------
def run_supervised(df):
    st.success("✅ Running Supervised Mode (using IsFraud labels).")

    target = "IsFraud"
    st.subheader("📊 Label Distribution")
    plot_pie(df[target], title="Fraud vs Legit Transactions")

    # Feature selection
    all_feats = [c for c in df.columns if c != target]
    st.sidebar.subheader("📊 Feature Selection")
    features = st.sidebar.multiselect("Select Feature Columns", all_feats, default=all_feats)

    if not features:
        st.error("⚠️ Please select at least one feature column.")
        st.stop()

    X = df[features].select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        st.error("⚠️ No numeric features found.")
        st.stop()

    y = df[target].astype(int)

    # Split + scale + balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_s, y_train)

    # Model choice
    st.sidebar.subheader("🤖 Choose Model")
    model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")

    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    # LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_res,
        feature_names=X.columns.tolist(),
        class_names=["Legit", "Fraud"],
        mode="classification"
    )

    # Evaluation
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Model Evaluation")
        st.code(classification_report(y_test, y_pred), language="text")
        st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.3f}")

    with col2:
        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    # ROC
    st.subheader("📉 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Precision-Recall
    st.subheader("📊 Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color="purple", linewidth=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend(loc="lower left"); ax.grid(True)
    st.pyplot(fig)

    # Feature Importance
    if model_choice == "Random Forest":
        st.subheader("🔑 Feature Importance")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=importances.index, ax=ax)
        st.pyplot(fig)

    # Try a new transaction
    st.subheader("🧪 Test a New Transaction")
    with st.expander("Enter Transaction Details"):
        input_vals = []
        cols = st.columns(3)
        for idx, col in enumerate(X.columns):
            cmin, cmax, cmean = float(X[col].min()), float(X[col].max()), float(X[col].mean())
            with cols[idx % 3]:
                v = st.number_input(f"{col}", cmin, cmax, cmean)
                input_vals.append(v)

        if st.button("🔎 Predict Fraud?"):
            scaled = scaler.transform([input_vals])
            pred = model.predict(scaled)[0]
            prob = model.predict_proba(scaled)[0][1]

            if pred == 1:
                st.error(f"🚨 Prediction: Fraudulent Transaction (Probability: {prob:.2f})")
            else:
                st.success(f"✅ Prediction: Legit Transaction (Probability: {prob:.2f})")

            st.subheader("🔍 LIME Explanation")
            exp = lime_explainer.explain_instance(np.array(scaled[0]), model.predict_proba, num_features=10)
            st.write(exp.as_list())
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)

            lime_results = exp.as_list()
            top_reasons = [f"{feat} ({'↑ Fraud' if weight > 0 else '↓ Legit'})"
                           for feat, weight in lime_results[:3]]
            st.info("This decision was mainly influenced by:\n" + " • " + "\n • ".join(top_reasons))

# -----------------------------
# Run Chosen Mode
# -----------------------------
if mode == "unsupervised":
    run_unsupervised(data)
else:
    run_supervised(data)
