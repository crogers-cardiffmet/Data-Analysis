
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import load_data

@st.cache_resource
def load_model_and_scaler():
    rf     = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return rf, scaler

@st.cache_data
def load_test_set():
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").squeeze()
    return X_test, y_test

def app():
    st.title("Modeling & Prediction")

    df      = load_data()
    rf, sc  = load_model_and_scaler()
    X_test, y_test = load_test_set()  

    X_test_s = sc.transform(X_test)
    y_pred   = rf.predict(X_test_s)

    # Metrics
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    st.subheader("Model Performance")
    st.write(f"- **MSE:**  {mse:.2f}")
    st.write(f"- **RMSE:** {rmse:.2f}")
    st.write(f"- **R²:**   {r2:.3f}")

    # Feature importance
    features = ['PM10','SO2','NO2','CO','O3']
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    st.subheader("Feature Importance")
    st.bar_chart(imp)

    # Actual vs predicted chart
    st.subheader("Actual vs Predicted PM2.5")
    comp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.line_chart(comp.reset_index(drop=True))

    # Residuals distribution
    st.subheader("Residuals Distribution")
    resid = y_test - y_pred
    fig, ax = plt.subplots()
    sns.histplot(resid, kde=True, ax=ax)
    st.pyplot(fig)

    # User prediction
    st.subheader("Make Your Own Prediction")
    user = {}
    for f in features:
        user[f] = st.number_input(f, float(df[f].min()), float(df[f].max()), float(df[f].median()))
    if st.button("Predict"):
        X_new = sc.transform(pd.DataFrame([user]))
        out   = rf.predict(X_new)[0]
        st.success(f"Predicted PM2.5: {out:.2f} µg/m³")
