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
    roc_curve
)
from imblearn.over_sampling import SMOTE

# ==============================
# Streamlit App Layout
# ==============================
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("---")

# Intro section
st.markdown("""
### ℹ️ About This Dashboard
This interactive dashboard demonstrates how **machine learning models** can detect fraudulent credit card transactions.  

- The dataset contains **100,000 simulated transactions**.  
- Each transaction has details like **amount, type, merchant, and location**.  
- Fraudulent transactions are rare (~1%), so we use **SMOTE** to balance the dataset.  

👉 Use the sidebar to choose features and models, explore results, and even test your own transaction!
""")

# Sidebar
st.sidebar.header("⚙️ Settings")

# Load dataset
DATA_PATH = "credit_card_fraud_dataset(100k transactions).csv"
data = pd.read_csv(DATA_PATH)

# ==============================
# Dataset Preview
# ==============================
st.subheader("📂 Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

# Fraud distribution pie chart
fraud_counts = data['IsFraud'].value_counts()
fig, ax = plt.subplots()
ax.pie(
    fraud_counts,
    labels=["Legit", "Fraud"],
    autopct='%1.1f%%',
    colors=["#4CAF50", "#FF5252"],
    startangle=90
)
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
X = data[features].select_dtypes(include=[np.number])
if X.shape[1] == 0:
    st.error("⚠️ No numeric features available. Please select numeric columns for modeling.")
    st.stop()

y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# ==============================
# Model Selection
# ==============================
st.sidebar.subheader("🤖 Choose Model")
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

model.fit(X_train_res, y_train_res)

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ==============================
# Dashboard Sections
# ==============================
col1, col2 = st.columns(2)

# Column 1 - Metrics
with col1:
    st.subheader("📈 Model Evaluation")
    st.text("Classification Report")
    st.code(classification_report(y_test, y_pred), language="text")
    st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.3f}")

# Column 2 - Confusion Matrix
with col2:
    st.subheader("🔍 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ROC Curve
st.subheader("📉 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, y_prob):.3f}")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
st.pyplot(fig)

# Feature Importance
if model_choice == "Random Forest":
    st.subheader("🔑 Feature Importance")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=importances.index, ax=ax)
    st.pyplot(fig)

# ==============================
# Test New Transaction
# ==============================
st.subheader("🧪 Test a New Transaction")

with st.expander("Enter Transaction Details"):
    input_data = []
    cols = st.columns(3)  # 3 inputs per row

    for idx, col in enumerate(X.columns):
        col_min, col_max, col_mean = float(X[col].min()), float(X[col].max()), float(X[col].mean())
        with cols[idx % 3]:
            val = st.number_input(f"{col}", col_min, col_max, col_mean)
            input_data.append(val)

    if st.button("🔎 Predict Fraud?"):
        input_scaled = scaler.transform([input_data])
        result = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if result == 1:
            st.error(f"🚨 Prediction: Fraudulent Transaction (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Prediction: Legit Transaction (Probability: {prob:.2f})")
