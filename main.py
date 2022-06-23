import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pandas import DataFrame

from Ticker import Ticker
from ModelCreator import ModelCreator

pd.set_option('display.max_colwidth', None)


class MainApp:
    models = ["LSTM", "SVM"]

    def __init__(self):

        # fields for data exchange
        self.stock = ""
        self.periodToFetch = 0
        self.fetchOptions = {}
        self.model = ""
        self.ticker = 0

        # fields for state persistance
        self.dataFetched = None
        self.graph = None

        # load state of main page
        self.loadMainStage()


    def loadMainStage(self):
        if 'fetchedData' in st.session_state.keys():
            self.__dict__ = st.session_state.fetchedData.__dict__

            st.write(f"Fetched {self.stock} data")
            st.dataframe(self.dataFetched[self.fetchOptions])
            st.plotly_chart(self.graph)


    # Fetching methods
    def selectStock(self):
        helpValue = "Model will be generated based on data fetched from Yahoo! Finance, " \
                    "based on a ticker name provided in a input."
        self.stock = st.text_input("Stock ticker:", value="NFLX", help=helpValue)
        return self.stock

    def selectPeriod(self):
        self.periodToFetch = st.slider("Days to fetch from data source:",
                                       min_value=1, max_value=365*2, value=180)
        return self.periodToFetch

    def selectFetchFilter(self):
        labels = ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits")
        with st.expander("Select filter"):
            self.fetchOptions = {label: st.checkbox(label, value=True) for label in labels}
            self.fetchOptions = [k for k in self.fetchOptions.keys() if self.fetchOptions.get(k)]
        return self.fetchOptions



    # Model methods
    def selectModel(self):
        self.model = st.selectbox("Select prediction model:", MainApp.models)
        return self.model

    def selectionModelExpander(self):
        if self.model == "LSTM":
            layers = ["LSTM", "Dense"]
            epochs = st.number_input("Epochs:", step=1, min_value=1)
            sender = ""
            with st.form("Add layer"):
                layerColumns = st.columns([1, 2])
                layer = layerColumns[0].selectbox("Layer:", layers)
                neurons = layerColumns[1].number_input("Neurons:", step=1, min_value=1)
                if st.form_submit_button("Add layer"):
                    sender = f"{layer} {neurons}\n"
            if "concat" not in st.session_state:
                st.session_state.concat = ""
            text = st.text_area("Layers parser", value=st.session_state.concat + sender)
            st.session_state.concat = text

    def selectDenseLayers(self):
        numberOfDense = int(st.number_input("Dense layers", min_value=1, max_value=10, step=1))
        neuronsOfLayers = st.text_input("Num. of neurons for Dense", autocomplete="1", placeholder="ex. 1")
        return 1

    def selectLSTMLayers(self):
        numberOfLSTMLayers = int(st.number_input("LSTM layers", min_value=1, max_value=10, step=1, value=2))
        neuronsOfLayers = st.text_input("Num. of neurons for LSTM", autocomplete="50 20", placeholder="ex. 50 20")
        return 1


    # sideBar creator
    def createSideBar(self):
        fetchSubmitted = False
        modelSubmitted = False
        timestep = 0

        # load state of fields

        with st.sidebar:
            st.write("Fetch stock data")
            with st.form("stockForm"):
                self.selectStock()
                self.selectPeriod()
                self.selectFetchFilter()
                fetchSubmitted = st.form_submit_button(label="Fetch Data")

            st.write("Create prediction")
            self.selectModel()
            self.selectionModelExpander()

            # with st.form("modelForm"):
            #     if self.model == "LSTM":
            #         timestep = st.slider("Timesteps:", min_value=1, max_value=100, step=1, value=60)
            #         self.selectLSTMLayers()
            #         self.selectDenseLayers()
            #     modelSubmitted = st.form_submit_button(label="Predict")

        if fetchSubmitted:
            self.drawMainStockData()
        if modelSubmitted:
            output = self.predictStock(timestep)



    # fetched Data drawer
    def drawMainStockData(self):
        self.ticker = Ticker(self.stock)
        self.dataFetched = self.ticker.getData(self.periodToFetch)
        self.graph = px.line(
            self.dataFetched["Close"],
            x=self.dataFetched.index,
            y="Close",
            markers=True,
            title=f"{self.stock} closing price line chart"
        )

        # save state
        st.session_state["fetchedData"] = self
        # rerun
        st.experimental_rerun()
        return 1

    def predictStock(self, timestep):
        self.__dict__ = st.session_state.fetchedData.__dict__ # copy of fields
        modelCreator = ModelCreator(self.dataFetched)
        st.plotly_chart(modelCreator.createLSTMPrediction(timestep))
        return 0


if __name__ == '__main__':
    app = MainApp()
    app.createSideBar()
