import streamlit as st
from multiapp import MultiApp

import data_overview
import eda
import modeling_prediction

app = MultiApp()

app.add_app("Data Overview", data_overview.app)
app.add_app("EDA", eda.app)
app.add_app("Modeling & Prediction", modeling_prediction.app)

app.run()