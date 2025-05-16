import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests

from data_loader import load_data
from sklearn.metrics import mean_squared_error, r2_score

S3_BUCKET = "air-quality-data-cardiff"
MODEL_KEY = "rf_model.pkl"
SCALER_KEY = "scaler.pkl"
XTEST_KEY = "X_test.csv"
YTEST_KEY = "y_test.csv"

BASE_URL = f"https://rf-model-cardiff.s3.eu-north-1.amazonaws.com/"

@st.cache_resource
def load_model_and_scaler():
    """Download (if needed) and load RF model + scaler from S3."""
    if not os.path.exists(MODEL_KEY):
        r = requests.get(BASE_URL + MODEL_KEY); r.raise_for_status()
        open(MODEL_KEY, "wb").write(r.content)
    if not os.path.exists(SCALER_KEY):
        r = requests.get(BASE_URL + SCALER_KEY); r.raise_for_status()
        open(SCALER_KEY, "wb").write(r.content)

    rf     = joblib.load(MODEL_KEY)
    scaler = joblib.load(SCALER_KEY)
    return rf, scaler

@st.cache_data
def load_test_set():
    """Download (if needed) and load X_test / y_test from S3."""
    if not os.path.exists(XTEST_KEY):
        r = requests.get(BASE_URL + XTEST_KEY); r.raise_for_status()
        open(XTEST_KEY, "wb").write(r.content)
    if not os.path.exists(YTEST_KEY):
        r = requests.get(BASE_URL + YTEST_KEY); r.raise_for_status()
        open(YTEST_KEY, "wb").write(r.content)

    X_test = pd.read_csv(XTEST_KEY)
    y_test = pd.read_csv(YTEST_KEY).squeeze()
    return X_test, y_test

def app():
    st.title("Modeling & Prediction")

    df = load_data()

    ext_ints = [
        c for c in df.columns
        if pd.api.types.is_integer_dtype(df[c].dtype) and df[c].isnull().any()
    ]
    for col in ext_ints:
        df[col] = df[col].astype("float64")

    # Snapshot
    st.subheader("Data Snapshot")
    st.table(df.head())
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # 2) Testing
    rf, scaler   = load_model_and_scaler()
    X_test, y_test = load_test_set()

    # 3) Scale & predict
    X_test_s = scaler.transform(X_test)
    y_pred   = rf.predict(X_test_s)

    # 4) Metrics
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    st.subheader("Test Set Performance")
    st.write(f"- **MSE:**  {mse:.2f}")
    st.write(f"- **RMSE:** {rmse:.2f}")
    st.write(f"- **R²:**   {r2:.3f}")

    # Feature importance
    features = X_test.columns.tolist()
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    st.subheader("Feature Importance")
    st.bar_chart(imp)

    # Actual vs. predicted
    st.subheader("Actual vs. Predicted PM2.5")
    comp = pd.table({"Actual": y_test, "Predicted": y_pred})
    st.line_chart(comp.reset_index(drop=True))

    # Residuals
    st.subheader("Residuals Distribution")
    resid = y_test - y_pred
    fig, ax = plt.subplots()
    sns.histplot(resid, kde=True, ax=ax)
    st.pyplot(fig)

    # 8) Custom prediction
    st.subheader("Make Your Own Prediction")
    user_inputs = {}
    for f in features:
        user_inputs[f] = st.number_input(f, float(df[f].min()), float(df[f].max()), float(df[f].median()))
    if st.button("Predict"):
        X_new = scaler.transform(pd.DataFrame([user_inputs]))
        out   = rf.predict(X_new)[0]
        st.success(f"Predicted PM2.5: {out:.2f} µg/m³")

