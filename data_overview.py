
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data

def app():
    st.title("Data Overview")
    df = load_data()

    # Basic info
    st.subheader("Sample & Shape")
    st.dataframe(df.head())
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Date filter
    if 'date' in df.columns:
        st.subheader("Filter by Date")
        min_date, max_date = st.date_input(
            "Date range",
            value=(df['date'].min().date(), df['date'].max().date())
        )
        mask = (df['date'].dt.date >= min_date) & (df['date'].dt.date <= max_date)
        df_filtered = df.loc[mask]
        st.write(f"Showing {df_filtered.shape[0]} rows between {min_date} and {max_date}")
        st.dataframe(df_filtered.head())

    # Show missing values 
    st.subheader("Missing Values")
    missing = pd.DataFrame({
        'count': df.isna().sum(),
        '%': 100*df.isna().mean()
    }).query("count>0")
    st.dataframe(missing.style.background_gradient("Reds"))
    st.subheader("Data types")
    st.write(df.dtypes)

    # Station and wind direction counts
    st.subheader("Station Counts")
    st.bar_chart(df['station'].value_counts())
    st.subheader("Wind Direction Counts")
    st.bar_chart(df['wd'].value_counts())

    # Correlation matrix
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_with_pm25 = numeric_df.corr()['PM2.5'].sort_values(ascending=False).drop('PM2.5')
    st.table(corr_with_pm25.to_frame(name="Correlation"))

    # PM2.5 distribution
    st.subheader("PM2.5 Distribution")
    fig, ax = plt.subplots(figsize=(6,3))
    sns.histplot(df['PM2.5'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
