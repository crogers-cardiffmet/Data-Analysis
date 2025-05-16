# in data_loader.py
import streamlit as st
import pandas as pd

@st.cache_data(show_spinner=False)
def load_data():
    station_files = {
      "Aotizhongxin":   "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_Data_Aotizhongxin_20130301-20170228.csv",
      "Changping":      "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_Data_Changping_20130301-20170228.csv",
      "Dingling":       "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_Data_Dingling_20130301-20170228.csv",
      "Wanshouxigong":  "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_Data_Wanshouxigong_20130301-20170228.csv",
    }
    dfs = []
    for name, url in station_files.items():
        try:
            df_site = pd.read_csv(url)
        except Exception as e:
            st.error(f"Failed to download {name} from\n{url}\n\n{e}")
            raise
        df_site["station"] = name
        dfs.append(df_site)
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(
        df[["year","month","day","hour"]].astype(str).agg("-".join, axis=1),
        format="%Y-%m-%d-%H"
    )
    return df

