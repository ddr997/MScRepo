import time

import streamlit as st
import pandas as pd
import numpy as np
from Ticker import Ticker
pd.set_option('display.max_colwidth', None)

class MainApp:
    models = ["LSTM", "SVM"]

    def __init__(self):
        self.stock = ""
        self.period = 0
        self.model = ""
        self.fetchOptions = {}

        self.ticker = None
        self.selectFilter = list()

    def selectStockTicker(self):
        helpValue = "Model will be generated based on data fetched from Yahoo! Finance, " \
                    "based on a ticker name provided in a input."
        return st.text_input("Stock ticker:", help=helpValue)

    def selectPeriod(self):
        return st.slider("Period to fetch from data source (days):", min_value=1, max_value=90)

    def selectModel(self):
        return st.selectbox("Select prediction model:", MainApp.models)

    def selectFetchFilter(self):
        labels = ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits")
        with st.expander("Select filter"):
            self.fetchOptions = {label: st.checkbox(label, value=True) for label in labels}
        return self.fetchOptions

    def createSideBar(self):
        fetchSubmitted = False

        with st.sidebar:
            with st.form("stockForm"):
                st.write("Fetch stock data")
                self.stock = self.selectStockTicker()
                self.period = self.selectPeriod()
                self.selectFetchFilter()
                fetchSubmitted = st.form_submit_button(label="Fetch Data")

            with st.form("modelForm"):
                st.write("Create model")
                self.model = self.selectModel()
                modelSubmitted = st.form_submit_button(label="Create data")

        if fetchSubmitted:
            self.drawMainStockData()

    def drawMainStockData(self):
        self.ticker = Ticker(self.stock)
        return st.dataframe(
            self.ticker.getData(self.period)
        )


if __name__ == '__main__':
    app = MainApp()
    app.createSideBar()
    # chart_data = pd.DataFrame(
    #     np.random.randn(20, 3),
    #     columns=['a', 'b', 'c'])
    #
    # st.line_chart(chart_data)
