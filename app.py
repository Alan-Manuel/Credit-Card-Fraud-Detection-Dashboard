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
import lime
import lime.lime_tabular

# --- Streamlit Layout ---
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("---")
st.markdown("""
### ℹ️ About This Dashboard
Detect fraudulent credit card transactions using machine learning.  
We use **SMOTE** to handle imbalance and tune thresholds for recall and ROC-AUC.
""")

st.sidebar.header("⚙️ Settings")

# --- Upload or Default CSV ---
st.sidebar.subheader("📁 Upload CSV Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

try:
    if uploaded_file is not None:
        st.sidebar.success("✅ File uploaded successfully!")
        data = pd.read_csv(uploaded_file)
    else:
        st.sidebar.info("ℹ️ No file uploaded. Using default dataset.")
        DATA_PATH = "credit_card_fraud_dataset.csv"
        data = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

if "IsFraud" not in data.columns:
    st.error("⚠️ The dataset must contain a target column named 'IsFraud'.")
    st.stop()

# --- Data Overview ---
st.subheader("📂 Dataset Overview")
st.write(f"**Rows:** {data.shape[0]} | **Columns:** {data.shape[1]}")
st.dataframe(data.head(10), width="stretch")

fraud_counts = data["IsFraud"].value_counts()
fig, ax = plt.subplots()
ax.pie(fraud_counts, labels=["Legit", "Fraud"], autopct="%1.1f%%",
       colors=["#4CAF50", "#FF5252"], startangle=90)
st.pyplot(fig)

# --- Feature Selection ---
target = "IsFraud"
features = st.sidebar.multiselect(
    "📊 Select Feature Columns",
    [c for c in data.columns if c != target],
    default=[c for c in data.columns if c != target]
)
if not features:
    st.error("⚠️ Please select at least one feature column.")
    st.stop()

X = pd.get_dummies(data[features], drop_first=True)
y = data[target]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Balance the Dataset ---
try:
    smote = SMOTE(random_state=42, sampling_strategy=0.2)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
except Exception as e:
    st.warning(f"⚠️ SMOTE skipped: {e}")
    X_train_res, y_train_res = X_train_scaled, y_train

# --- Model Selection ---
st.sidebar.subheader("🤖 Choose Model")
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
else:
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

model.fit(X_train_res, y_train_res)

# --- Prediction & Threshold ---
st.sidebar.subheader("🎚️ Prediction Threshold")
threshold = st.sidebar.slider("Fraud Probability Cutoff", 0.0, 1.0, 0.5, 0.05)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob > threshold).astype(int)

# --- Evaluation ---
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

# --- Curves ---
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

# --- Manual Transaction Testing ---
st.subheader("🧪 Test a New Transaction")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_res,
    feature_names=X.columns,
    class_names=["Legit", "Fraud"],
    mode="classification"
)

with st.expander("Enter Transaction Details"):
    input_data = []
    cols = st.columns(3)
    for idx, col in enumerate(X.columns[:9]):
        col_min, col_max, col_mean = float(X[col].min()), float(X[col].max()), float(X[col].mean())
        with cols[idx % 3]:
            val = st.number_input(f"{col}", col_min, col_max, col_mean)
            input_data.append(val)
    if st.button("🔎 Predict Fraud?"):
        if len(input_data) < X_train.shape[1]:
            input_data += [0] * (X_train.shape[1] - len(input_data))
        input_scaled = scaler.transform([input_data])
        result = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        if result == 1:
            st.error(f"🚨 Fraudulent (Prob: {prob:.2f})")
        else:
            st.success(f"✅ Legitimate (Prob: {prob:.2f})")
