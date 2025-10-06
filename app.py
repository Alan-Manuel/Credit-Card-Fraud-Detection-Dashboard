import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier  

import lime
import lime.lime_tabular

# ==============================
# Streamlit App Layout
# ==============================
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("---")

st.markdown("""
### ℹ️ About This Dashboard
Detect fraudulent credit card transactions using multiple ML models.  
The dataset is heavily imbalanced (~1% fraud), so we use **SMOTE**, **XGBoost**, and **threshold tuning** to improve recall and ROC-AUC.
""")

st.sidebar.header("⚙️ Settings")

# ==============================
# CSV Upload Feature
# ==============================
st.sidebar.subheader("📁 Upload CSV Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    st.sidebar.success("✅ File uploaded successfully!")
    data = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("ℹ️ No file uploaded. Using default dataset.")
    DATA_PATH = "credit_card_fraud_dataset(100k transactions).csv"
    data = pd.read_csv(DATA_PATH)

if "IsFraud" not in data.columns:
    st.error("⚠️ The dataset must contain a target column named 'IsFraud'.")
    st.stop()

# ==============================
# Dataset Preview
# ==============================
st.subheader("📂 Dataset Overview")
st.write(f"**Rows:** {data.shape[0]} | **Columns:** {data.shape[1]}")
st.dataframe(data.head(10), use_container_width=True)

# Fraud distribution pie chart
fraud_counts = data['IsFraud'].value_counts()
fig, ax = plt.subplots()
ax.pie(fraud_counts, labels=["Legit", "Fraud"], autopct='%1.1f%%',
       colors=["#4CAF50", "#FF5252"], startangle=90)
ax.set_title("Fraud vs Legit Transactions")
st.pyplot(fig)

# ==============================
# Target and Features
# ==============================
target = "IsFraud"
features = st.sidebar.multiselect(
    "📊 Select Feature Columns",
    [c for c in data.columns if c != target],
    default=[c for c in data.columns if c != target]
)
if not features:
    st.error("⚠️ Please select at least one feature column.")
    st.stop()

# ==============================
# Data Preprocessing
# ==============================
X = data[features]
y = data[target]

# ✅ Encode categoricals
X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Handle imbalance with tuned SMOTE
smote = SMOTE(random_state=42, sampling_strategy=0.2)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# ==============================
# Model Selection
# ==============================
st.sidebar.subheader("🤖 Choose Model")
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest", "XGBoost"])

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
elif model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
else:
    # ✅ XGBoost for stronger performance
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_train_res) / sum(y_train_res)),
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

model.fit(X_train_res, y_train_res)

# ==============================
# Prediction & Threshold Tuning
# ==============================
st.sidebar.subheader("🎚️ Prediction Threshold")
threshold = st.sidebar.slider("Fraud Probability Cutoff", 0.0, 1.0, 0.5, 0.05)

y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob > threshold).astype(int)

# ==============================
# Model Evaluation
# ==============================
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Model Evaluation")
    st.text("Classification Report")
    st.code(classification_report(y_test, y_pred, digits=3), language="text")
    st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.3f}")

with col2:
    st.subheader("🔍 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ==============================
# ROC Curve
# ==============================
st.subheader("📉 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, y_prob):.3f}")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.legend(loc="lower right")
st.pyplot(fig)

# ✅ Precision-Recall Curve
st.subheader("🎯 Precision-Recall Curve")
prec, rec, thr = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots()
ax.plot(rec, prec)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
st.pyplot(fig)

# Feature Importance (RF & XGB)
if model_choice in ["Random Forest", "XGBoost"]:
    st.subheader("🔑 Feature Importance")
    importances = (
        model.feature_importances_ if model_choice == "XGBoost"
        else model.feature_importances_
    )
    importance_df = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=importance_df, y=importance_df.index, ax=ax)
    st.pyplot(fig)

# ==============================
# LIME + Manual Transaction
# ==============================
st.subheader("🧪 Test a New Transaction")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_res,
    feature_names=X.columns,
    class_names=['Legit', 'Fraud'],
    mode='classification'
)

with st.expander("Enter Transaction Details"):
    input_data = []
    cols = st.columns(3)
    for idx, col in enumerate(X.columns[:9]):  # limit to key inputs
        col_min, col_max, col_mean = float(X[col].min()), float(X[col].max()), float(X[col].mean())
        with cols[idx % 3]:
            val = st.number_input(f"{col}", col_min, col_max, col_mean)
            input_data.append(val)

    if st.button("🔎 Predict Fraud?"):
        input_scaled = scaler.transform([input_data + [0]*(X_train.shape[1]-len(input_data))])
        result = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        if result == 1:
            st.error(f"🚨 Fraudulent (Prob: {prob:.2f})")
        else:
            st.success(f"✅ Legitimate (Prob: {prob:.2f})")
