# data_loader.py

import streamlit as st
import pandas as pd
import os

@st.cache_data(show_spinner=False)
def load_data():
    here     = os.path.dirname(__file__)
    zip_name = "combined_output.zip"           # or "data/combined_output.zip" if you placed it in a data/ subfolder
    zip_path = os.path.join(here, zip_name)

    # DEBUG
    st.write(f"🔍 Looking for ZIP at: `{zip_path}`")
    st.write(f"🗂️  Files in this folder: {os.listdir(here)}")
    if not os.path.exists(zip_path):
        st.error(f"❌ ZIP file *not* found at that path!")
        raise FileNotFoundError(f"Could not find {zip_name} in {here}")

    st.write("✅ ZIP found, loading now…")
    df = pd.read_csv(zip_path, compression="zip")

    if {"year","month","day","hour"}.issubset(df.columns):
        df["date"] = pd.to_datetime(
            df[["year","month","day","hour"]]
              .astype(str)
              .agg("-", axis=1),
            format="%Y-%m-%d-%H"
        )
    return df



