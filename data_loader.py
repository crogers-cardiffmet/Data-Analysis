# data_loader.py

import streamlit as st
import pandas as pd

@st.cache_data(show_spinner=False)
def load_data():
    # URLs for your four selected sites
    station_files = {
        "Aotizhongxin":   "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_Data_Aotizhongxin_20130301-20170228.csv",
        "Changping":      "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_Data_Changping_20130301-20170228.csv",
        "Dingling":       "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_Data_Dingling_20130301-20170228.csv",
        "Wanshouxigong":  "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_Data_Wanshouxigong_20130301-20170228.csv",
    }

    # Download each CSV and tag it by station
    dfs = []
    for station, url in station_files.items():
        df_site = pd.read_csv(url)
        df_site["station"] = station
        dfs.append(df_site)

    # Concatenate into one DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # Combine year/month/day/hour into a datetime
    df["date"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]]
          .astype(str)
          .agg("-", axis=1),
        format="%Y-%m-%d-%H"
    )

    return df
