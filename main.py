import pandas as pd
import plotly.express as px
import streamlit as st

from StateData import StateData
from ModelCreator import ModelCreator
from Ticker import Ticker
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
            # if "plotPred" in st.session_state.keys():
            #     st.plotly_chart(st.session_state.plotPred)
            #     col1, col2 = st.columns(2)
            #     col1.metric("RMSE", st.session_state.rmse)
            #     col2.metric("MAPE", st.session_state.mape)

    # Model methods
    # def selectModel(self):
    #     self.model = st.selectbox("Select prediction model:", MainApp.models)
    #     return self.model

    # def selectionModelExpander(self):
    #     if self.model == "LSTM":
    #         layers = ["LSTM", "Dense", "Dropout"]
    #         networkColumns = st.columns([1, 1])
    #         epochs = networkColumns[0].number_input("Epochs:", step=1, min_value=1, value=10)
    #         batchSize = networkColumns[1].number_input("Batch size:", step=1, min_value=1, value=32)
    #         windowSize = st.slider("Window size:", min_value=1, max_value=120, step=1, value=20)
    #         sender = ""
    #
    #         with st.form("Add layer"):
    #             layerColumns = st.columns([1, 1])
    #             layer = layerColumns[0].selectbox("Layer:", layers)
    #             neurons = layerColumns[1].number_input("Neurons:", step=1, min_value=1)
    #             # returnSequence = st.selectbox("Return seq.", [True, False])
    #             if st.form_submit_button("Add layer"):
    #                 sender = f"\n{layer} {neurons}"
    #
    #         if "concat" not in st.session_state:
    #             st.session_state.concat = "LSTM 96\nDropout 0.2\nLSTM 96\nDropout 0.2\nLSTM 96\nDropout 0.2\nDense 1"
    #         text = st.text_area("Layers parser", value=st.session_state.concat + sender)
    #         st.session_state.concat = text
    #
    #         if st.button("Create prediction"):
    #             mc = ModelCreator(self.dataFetched)
    #             mc.parseLayers(text)
    #             st.session_state.plotPred, st.session_state.rmse, st.session_state.mape \
    #                 = mc.createLSTMPrediction(epochs, batchSize, windowSize)
    #             return 2
    #
    #     if self.model == "Linear Regression":
    #         if st.button("Create prediction"):
    #             mc = ModelCreator(self.dataFetched)
    #             mc.createLinearRegressionPrediction()
    #         return 1

    # sideBar creator
    def createSideBar(self):
        # modelSubmitted = False

        StockMenu(self.stateData)
        IndicatorsMenu(self.stateData)
        ModelsMenu(self.stateData)
        # with st.sidebar:
        #     plotPred = self.selectionModelExpander()

        # if stockMenu.fetchSubmitted:
        #     self.drawMainStockData()
        # if plotPred == 2:
        #     st.experimental_rerun()
        # if plotPred == 1:
        #     print(plotPred)

if __name__ == '__main__':
    app = MainApp()
    app.createSideBar()
