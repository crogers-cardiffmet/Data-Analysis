import streamlit as st
import pandas as pd
import os
import gdown

@st.cache_data(show_spinner=False)
def load_data():
    # 1) Google Drive file ID
    file_id = "1_uCyJOFCmVoEj7Wc_Z_PE4Z9iuUe6e2BX"
    url     = f"https://drive.google.com/uc?id={file_id}"
    output  = "combined_output.csv"

    # 2) Download only if not already present
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    # 3) Read the CSV
    df = pd.read_csv(output)

    # 4) Rebuild your datetime column
    if {"year","month","day","hour"}.issubset(df.columns):
        df["date"] = pd.to_datetime(
            df[["year","month","day","hour"]]
              .astype(str).agg("-", axis=1),
            format="%Y-%m-%d-%H"
        )

    return df





