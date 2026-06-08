import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Breast Cancer AI System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Prediction System (AI Powered)")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("breast-cancer.csv")

df = load_data()

uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# clean dataset
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
if "id" in df.columns:
    df = df.drop("id", axis=1)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# target encoding (important for ML stability)
if df["diagnosis"].dtype == "object":
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ---------------- MODELS ----------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier()
}

results = {}
trained_models = {}

# ---------------- TRAIN ----------------
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    trained_models[name] = model

    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred)
    }

# ---------------- RESULTS TABLE ----------------
results_df = pd.DataFrame({
    k: {
        "Accuracy": v["accuracy"],
        "Precision": v["precision"],
        "Recall": v["recall"],
        "F1 Score": v["f1"]
    }
    for k, v in results.items()
}).T

st.subheader("📈 Model Comparison")
st.dataframe(results_df)

best_model_name = results_df["Accuracy"].idxmax()
best_model = trained_models[best_model_name]

st.success(f"🏆 Best Model: {best_model_name}")

predictions = best_model.predict(X_test)

# ---------------- METRICS ----------------
st.subheader("🏆 Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy_score(y_test, predictions)*100:.2f}%")
col2.metric("Precision", f"{precision_score(y_test, predictions)*100:.2f}%")
col3.metric("Recall", f"{recall_score(y_test, predictions)*100:.2f}%")
col4.metric("F1 Score", f"{f1_score(y_test, predictions)*100:.2f}%")

# ---------------- CONFUSION MATRIX ----------------
st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_test, predictions)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# ---------------- ROC CURVE ----------------
st.subheader("📊 ROC Curve")

if hasattr(best_model, "predict_proba"):
    probs = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], linestyle="--")
    ax2.legend()
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    st.pyplot(fig2)

# ---------------- CLASSIFICATION REPORT ----------------
st.subheader("📄 Classification Report")

report = classification_report(y_test, predictions)
st.text(report)

buffer = io.StringIO()
buffer.write(report)

st.download_button(
    "⬇ Download Report",
    buffer.getvalue(),
    file_name="report.txt"
)

# ---------------- PREDICTION ----------------
st.subheader("🧠 Predict New Patient")

input_data = []

cols = st.columns(3)

for i, col in enumerate(X.columns):
    with cols[i % 3]:
        value = col.number_input if False else st.number_input(col, value=0.0)
        input_data.append(value)

if st.button("🔍 Predict"):

    input_df = pd.DataFrame([input_data], columns=X.columns)

    prediction = best_model.predict(input_df)[0]
    confidence = np.max(best_model.predict_proba(input_df))

    if prediction == 1:
        st.error("⚠️ Malignant Tumor Detected")
    else:
        st.success("✅ Benign Tumor")

    st.write(f"Confidence: {confidence*100:.2f}%")

# ---------------- SAVE MODEL ----------------
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

st.success("✅ Model saved successfully")
