import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns


# Streamlit app DASHBOARD
st.title('Credit Card Fraud Detection Dashboard')

st.sidebar.header("Upload and Settings")

# Load dataset
DATA_PATH = "credit_card_fraud_dataset(100k transactions).csv"
data = pd.read_csv(DATA_PATH)

st.subheader("Dataset Preview")
st.write(data.head())

# Sidebar: Choose target & features
target = st.sidebar.selectbox("Select Target Variable", data.columns)
features = st.sidebar.multiselect(
    "Select feature columns",
    [c for c in data.columns if c != target],
    default=[c for c in data.columns if c != target]
)

if not features:
    st.error("⚠️ Please select at least one feature column.")
    st.stop()

# Define X and y
X = data[features]
y = data[target]

# Show class distribution
st.subheader("Target Variable Distribution")
st.write(y.value_counts())

# Check target validity
if y.nunique() < 2:
    st.error("❌ Target variable must have at least 2 unique classes (e.g., 0 and 1).")
    st.stop()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Sidebar model choice
st.sidebar.subheader("Choose Model")
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_res, y_train_res)
else:
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    model.fit(X_train_res, y_train_res)

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

st.subheader("Model Evaluation")
st.code(classification_report(y_test, y_pred), language="text")
st.text(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Predict new transaction
st.subheader("🔎 Test a New Transaction")
input_data = []
for col in features:
    val = st.number_input(
        f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean())
    )
    input_data.append(val)

if st.button("Predict Fraud?"):
    input_scaled = scaler.transform([input_data])
    result = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    st.write(f"**Prediction:** {'Fraudulent' if result == 1 else 'Legit'}")
    st.write(f"**Fraud Probability:** {prob:.2f}")

