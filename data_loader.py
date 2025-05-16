import os
import streamlit as st
import pandas as pd

@st.cache_data(show_spinner=False)
def load_data():
    url = "https://air-quality-data-cardiff.s3.eu-north-1.amazonaws.com/combined_output+(1).csv"
    
    fn = "combined_output.csv"
    if not os.path.exists(fn):
        df = pd.read_csv(url)
        df.to_csv(fn, index=False)
    else:
        df = pd.read_csv(fn)

    if {'year','month','day','hour'}.issubset(df.columns):
        df['date'] = pd.to_datetime(
            df[['year','month','day','hour']].astype(str)
              .apply(lambda row: '-'.join(row), axis=1),
            format='%Y-%m-%d-%H'
        )
    return df







