import streamlit as st
import pandas as pd

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(
        "data/combined_output.zip",    
        compression="zip"             
    )
    if {"year","month","day","hour"}.issubset(df.columns):
        df["date"] = pd.to_datetime(
            df[["year","month","day","hour"]]
              .astype(str)
              .agg("-", axis=1),
            format="%Y-%m-%d-%H"
        )

    return df


