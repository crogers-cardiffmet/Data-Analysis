import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data_loader import load_data

# Defining wind direction order for consistency
wind_dir_order = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

def app():
    st.title("Exploratory Data Analysis")

    df = load_data()
    ext_ints = [
        c for c in df.columns
        if pd.api.types.is_integer_dtype(df[c].dtype) and df[c].isnull().any()
    ]
    for col in ext_ints:
        df[col] = df[col].astype("float64")

    st.subheader("Sample & Shape")
    st.dataframe(df.head())
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    
    numeric = ['PM2.5','PM10','SO2','NO2','CO','O3']

    # Time range & granularity
    st.subheader("Time Range & Granularity")
    min_date, max_date = st.date_input(
    "Select a date range",
    value=(df['date'].min().date(), df['date'].max().date())
    )
    mask = (df['date'].dt.date >= min_date) & (df['date'].dt.date <= max_date)
    df_slice = df.loc[mask]

    gran = st.radio("Resample frequency", ['D','W','M'], index=2, format_func=lambda x: {'D':'Daily','W':'Weekly','M':'Monthly'}[x])
    ts = df_slice.set_index('date')[numeric].resample(gran).mean()
    st.line_chart(ts)

    # Distributions
    st.subheader("Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(15,6))
    for i, col in enumerate(numeric):
        ax = axes.flatten()[i]
        sns.histplot(df_slice[col].dropna(), kde=True, ax=ax)
        ax.set_title(col)
    st.pyplot(fig)

    # Boxplots
    st.subheader("Boxplots")
    fig, axes = plt.subplots(2, 3, figsize=(15,6))
    for i, col in enumerate(numeric):
        ax = axes.flatten()[i]
        sns.boxplot(x=df_slice[col].dropna(), ax=ax, color='orange')
        ax.set_title(col)
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df_slice[numeric].corr()
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)

    # Pairwise scatter matrix
    st.subheader("Pairwise Scatter Matrix")
    cols = st.multiselect("Pick features for pairplot", numeric, default=numeric[:4])
    if len(cols) >= 2:
        sample = df_slice[cols].dropna().sample(min(len(df_slice), 2000), random_state=42)
        fig = sns.pairplot(sample, corner=True, plot_kws={'alpha':0.3})
        st.pyplot(fig)
    else:
        st.info("Select at least two features to see a pairplot.")

    # Seasonal distributions
    st.subheader("Seasonal Distributions")
    seasons = {12:'Winter',1:'Winter',2:'Winter',
               3:'Spring',4:'Spring',5:'Spring',
               6:'Summer',7:'Summer',8:'Summer',
               9:'Fall',10:'Fall',11:'Fall'}
    df_slice['season'] = df_slice['month'].map(seasons)
    sel_pollutant = st.selectbox("Pick a pollutant for season boxplot", numeric)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(x='season', y=sel_pollutant, data=df_slice, 
                order=['Winter','Spring','Summer','Fall'], ax=ax)
    ax.set_title(f"{sel_pollutant} by Season")
    st.pyplot(fig)

    # Custom scatter plot
    st.subheader("Custom Scatter Plot")
    xcol = st.selectbox("X axis", numeric, index=0)
    ycol = st.selectbox("Y axis", numeric, index=1)
    hue = st.selectbox("Color by", [None] + numeric)
    fig, ax = plt.subplots(figsize=(6,4))
    if hue:
        sc = ax.scatter(df_slice[xcol], df_slice[ycol], c=df_slice[hue], cmap='viridis', alpha=0.5)
        plt.colorbar(sc, ax=ax, label=hue)
    else:
        ax.scatter(df_slice[xcol], df_slice[ycol], alpha=0.5)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"{ycol} vs {xcol}")
    st.pyplot(fig)
