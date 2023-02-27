import pandas as pd
import streamlit as st

from StateData import StateData
from UI.central.Estimators import Estimators
from UI.side.IndicatorsMenu import IndicatorsMenu
from UI.side.ModelsMenu import ModelsMenu
from UI.side.StockMenu import StockMenu
from UI.central.PredictionPlot import PredictionPlot

pd.set_option('display.max_colwidth', None)
st.set_page_config(layout="wide")
st.markdown(
    f"""
    <style>
    .appview-container .main .block-container{{
        padding-top: {2}rem;
        }}
    .reportview-container .sidebar-content {{
        padding-top: {1}rem;
        }}
    .css-1gx893w.egzxvld2 {{
      margin-top: -75px;
    }}
    </style>
    """, unsafe_allow_html=True
)

class MainApp:
    models = ["LSTM", "Linear Regression"]
    def __init__(self):
        self.stateData = StateData()
        # load state of main page
        self.mainStage()

    def mainStage(self):
        if 'state' in st.session_state.keys():
            self.stateData = st.session_state.state
            st.write(f"Fetched {self.stateData.stock} data")
            st.dataframe(self.stateData.dataFrame.iloc[::-1])
            subPlotIndicatorName = st.selectbox("Select indicator to plot:", self.stateData.dataFrame.columns[4:])
            if subPlotIndicatorName in self.stateData.dataFrame.columns:
                self.stateData.plot.addIndicatorTrace(self.stateData.dataFrame, y=subPlotIndicatorName)
            st.plotly_chart(self.stateData.plot.fig, theme=None)
            if self.stateData.predictionDataFrame is not None:
                st.plotly_chart(
                    PredictionPlot(self.stateData.predictionDataFrame).getPlot()
                )
                Estimators(self.stateData)

    # sideBar creator
    def createSideBar(self):
        StockMenu(self.stateData)
        IndicatorsMenu(self.stateData)
        ModelsMenu(self.stateData)

if __name__ == '__main__':
    app = MainApp()
    app.createSideBar()
