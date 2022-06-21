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
        self.stock = ""
        self.period = 0
        self.model = ""
        self.fetchOptions = {}
        self.ticker = 0
        self.data = None

    # Fetching methods

    def selectStock(self):
        helpValue = "Model will be generated based on data fetched from Yahoo! Finance, " \
                    "based on a ticker name provided in a input."
        self.stock = st.text_input("Stock ticker:", value="NFLX", help=helpValue)
        return self.stock

    def selectPeriod(self):
        self.period = st.slider("Period to fetch from data source (days):",
                                min_value=1, max_value=365, value=90)
        return self.period

    def selectFetchFilter(self):
        labels = ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits")
        with st.expander("Select filter"):
            self.fetchOptions = {label: st.checkbox(label, value=True) for label in labels}
        return self.fetchOptions

    # Model methods

    def selectModel(self):
        self.model = st.selectbox("Select prediction model:", MainApp.models)
        return self.model

    def selectDenseLayers(self):
        numberOfDense = int(st.number_input("Dense layers", min_value=1, max_value=10, step=1))
        neuronsOfLayers = st.text_input("Num. of neurons for Dense", autocomplete="1", placeholder="ex. 1")
        return 1

    def selectLSTMLayers(self):
        numberOfLSTMLayers = int(st.number_input("LSTM layers", min_value=1, max_value=10, step=1, value=2))
        neuronsOfLayers = st.text_input("Num. of neurons for LSTM", autocomplete="50 20", placeholder="ex. 50 20")
        return 1

    def createSideBar(self):
        fetchSubmitted = False
        modelSubmitted = False

        timestep = 0

        with st.sidebar:
            st.write("Fetch stock data")
            with st.form("stockForm"):
                self.selectStock()
                self.selectPeriod()
                self.selectFetchFilter()
                fetchSubmitted = st.form_submit_button(label="Fetch Data")

            st.write("Create prediction")
            self.model = self.selectModel()
            with st.form("modelForm"):
                if self.model == "LSTM":
                    timestep = st.slider("Timesteps:", min_value=1, max_value=100, step=1, value=60)
                    self.selectLSTMLayers()
                    self.selectDenseLayers()
                modelSubmitted = st.form_submit_button(label="Predict")

        if fetchSubmitted:
            self.drawMainStockData()
        if modelSubmitted:
            output = self.predictStock(timestep)

    def drawMainStockData(self):
        self.ticker = Ticker(self.stock)
        selected = [k for k in self.fetchOptions.keys() if self.fetchOptions.get(k)]
        self.data = self.ticker.getData(self.period)

        st.write(f"Fetched {self.stock} data")
        st.dataframe(self.data[selected])

        fig = px.line(
            self.data["Close"],
            x=self.data.index,
            y="Close",
            markers=True,
            title=f"{self.stock} closing price line chart")
        st.plotly_chart(fig)

        st.session_state["fetchedData"] = self
        return 1

    def predictStock(self, timestep):
        self.__dict__ = st.session_state.fetchedData.__dict__ # copy of fields
        modelCreator = ModelCreator(self.data)
        st.plotly_chart(modelCreator.createLSTMPrediction(timestep))
        return 0

if __name__ == '__main__':
    app = MainApp()
    app.createSideBar()
