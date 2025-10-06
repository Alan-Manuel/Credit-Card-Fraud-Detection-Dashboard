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
    classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import lime
import lime.lime_tabular

# ==============================
# Streamlit App Layout
# ==============================
st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("---")

st.markdown("""
### ℹ️ About This Dashboard
Detect fraudulent credit card transactions using machine learning.

- Dataset contains **~100,000 transactions**.
- Fraudulent transactions are rare (~1%), so **SMOTE** is used for balance.
- Supports **Logistic Regression** and **Random Forest** models.
""")

# ==============================
# Sidebar Controls
# ==============================
st.sidebar.header("⚙️ Settings")

st.sidebar.subheader("📁 Upload CSV Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    st.sidebar.success("✅ File uploaded successfully!")
    data = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("ℹ️ No file uploaded. Using default dataset.")
    DATA_PATH = "credit_card_fraud_dataset.csv"
    data = pd.read_csv(DATA_PATH)

# ==============================
# Data Check
# ==============================
if "IsFraud" not in data.columns:
    st.error("⚠️ Dataset must contain a target column named 'IsFraud'.")
    st.stop()

# ==============================
# Dataset Preview
# ==============================
st.subheader("📂 Dataset Overview")
st.write(f"**Rows:** {data.shape[0]} | **Columns:** {data.shape[1]}")
st.dataframe(data.head(10), use_container_width=True)

# Fraud Distribution
fraud_counts = data["IsFraud"].value_counts()
fig, ax = plt.subplots()
ax.pie(
    fraud_counts,
    labels=["Legit", "Fraud"],
    autopct="%1.1f%%",
    colors=["#4CAF50", "#FF5252"],
    startangle=90
)
ax.set_title("Fraud vs Legit Transactions")
st.pyplot(fig)

# ==============================
# Feature Selection
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

X = data[features].select_dtypes(include=[np.number])
if X.shape[1] == 0:
    st.error("⚠️ No numeric features available for modeling.")
    st.stop()

y = data[target]

# ==============================
# Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==============================
# SMOTE (Before Scaling)
# ==============================
smote = SMOTE(random_state=42, sampling_strategy=1.0)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

st.sidebar.write(f"📈 Fraud Ratio After SMOTE: {y_train_res.mean():.2f}")

# ==============================
# Scale After SMOTE
# ==============================
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Model Selection
# ==============================
st.sidebar.subheader("🤖 Choose Model")
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=0.5,
        solver="lbfgs"
    )
else:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced_subsample"
    )

model.fit(X_train_res, y_train_res)

# ==============================
# Threshold Slider
# ==============================
st.sidebar.subheader("🎚️ Prediction Threshold")
threshold = st.sidebar.slider("Fraud Probability Cutoff", 0.0, 1.0, 0.25, 0.05)

# Predictions
y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob > threshold).astype(int)

# ==============================
# Evaluation
# ==============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Model Evaluation")
    st.code(classification_report(y_test, y_pred, digits=3), language="text")
    st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.3f}")

with col2:
    st.subheader("🔍 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

# ==============================
# ROC & Precision-Recall Curves
# ==============================
st.subheader("📉 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, y_prob):.3f}")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.legend(loc="lower right")
st.pyplot(fig)

st.subheader("🎯 Precision-Recall Curve")
prec, rec, _ = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots()
ax.plot(rec, prec)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
st.pyplot(fig)

# ==============================
# Feature Importance
# ==============================
if model_choice == "Random Forest":
    st.subheader("🔑 Feature Importance")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=importances.index, ax=ax)
    st.pyplot(fig)

# ==============================
# LIME Explainability
# ==============================
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_res,
    feature_names=X.columns,
    class_names=["Legit", "Fraud"],
    mode="classification"
)

# ==============================
# New Transaction Test
# ==============================
st.subheader("🧪 Test a New Transaction")

with st.expander("Enter Transaction Details"):
    input_data = []
    cols = st.columns(3)

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

        # --- LIME Explanation ---
        st.subheader("🔍 LIME Explanation (Top Features)")
        exp = lime_explainer.explain_instance(
            np.array(input_scaled[0]),
            model.predict_proba,
            num_features=10
        )
        st.write(exp.as_list())
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)

        # --- Simple Text Summary ---
        st.subheader("📝 Explanation in Simple Words")
        lime_results = exp.as_list()
        top_reasons = [
            f"{feat} ({'↑ Fraud' if weight > 0 else '↓ Legit'})"
            for feat, weight in lime_results[:3]
        ]
        st.info("This decision was mainly influenced by:\n" + " • " + "\n • ".join(top_reasons))
