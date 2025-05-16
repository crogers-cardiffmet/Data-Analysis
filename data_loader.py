import os
import streamlit as st
import pandas as pd

@st.cache_data(show_spinner=False)
def load_data():
    """
    Fetch the merged air‐quality CSV from S3 over HTTP,
    rebuild the datetime column, and return the DataFrame.
    """
    url = (
        "https://air-quality-data-cardiff.s3.eu-north-1.amazonaws.com/combined_output+(1).csv"
    )
    local_fn = "combined_output.csv"

    if not os.path.exists(local_fn):
        df = pd.read_csv(url)
        df.to_csv(local_fn, index=False)
    else:
        df = pd.read_csv(local_fn)

    needed = {"year", "month", "day", "hour"}
    if needed.issubset(df.columns):
        df["month2"] = df["month"].astype(int).astype(str).str.zfill(2)
        df["day2"]   = df["day"].astype(int).astype(str).str.zfill(2)
        df["hour2"]  = df["hour"].astype(int).astype(str).str.zfill(2)

        df["date_str"] = (
            df["year"].astype(str) + "-" +
            df["month2"] + "-" +
            df["day2"]   + "T" +
            df["hour2"] + ":00:00"
        )
        df["date"] = pd.to_datetime(
            df["date_str"], format="%Y-%m-%dT%H:%M:%S"
        )
    return df









