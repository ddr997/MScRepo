from StateData import StateData
import pandas as pd
import streamlit as st

class Estimators:
    def __init__(self, stateData : StateData):
        dfMetrics = pd.DataFrame.from_dict(
            stateData.metrics, orient="index", columns=["MSE", "MAPE", "MDE"]
        )
        st.dataframe(dfMetrics)