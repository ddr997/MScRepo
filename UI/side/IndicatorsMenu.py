import streamlit as st
import pandas as pd
from StateData import StateData
from indicators.IndicatorsFacade import IndicatorsFacade

class IndicatorsMenu:
    def __init__(self, stateData: StateData):
        self.submit = False
        self.stateData = stateData # used for clearing dataframe

        with st.sidebar:
            st.write("Select indicators to calculate")
            labels = IndicatorsFacade.indicatorsList
            with st.form("indicatorsForm"):
                with st.expander("Select filter"):
                    stateData.selectedIndicators = {label: st.checkbox(label, value=False) for label in labels}
                    stateData.selectedIndicators = [k for k in stateData.selectedIndicators.keys() if
                                                  stateData.selectedIndicators.get(k)]
                self.submit = st.form_submit_button(label="Calculate")
        if (self.submit):
            self.clearDataFrame(stateData)
            self.calculateSelectedIndicators(stateData)
            st.session_state["state"] = stateData
            st.experimental_rerun()


    def calculateSelectedIndicators(self, stateData: StateData):
        [IndicatorsFacade.map(k)(stateData.dataFrame) for k in stateData.selectedIndicators
         if k not in stateData.dataFrame.columns]
        print(stateData.dataFrame)
        return 0

    def clearDataFrame(self, stateData: StateData):
        stateData.dataFrame = stateData.cleanDataFrame.copy(deep=True)
        stateData.predictionDataFrame = pd.DataFrame(stateData.dataFrame["Close"])

