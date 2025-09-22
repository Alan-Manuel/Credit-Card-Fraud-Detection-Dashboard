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
    roc_curve, precision_score, recall_score, f1_score
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

# ===========

