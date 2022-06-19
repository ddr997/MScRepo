import time

import streamlit as st
import pandas as pd
import numpy as np
from DataFetcher import DataFetcher

class MainApp:
    models = ["LSTM", "SVM"]
    def __init__(self):
        self.model = "DEF"
        self.stock = "DEF"
        self.sidebar = st.sidebar

    def selectStockTicker(self):
        helpValue = "Model will be generated based on data fetched from Yahoo! Finance, " \
                    "based on a ticker name provided in a input."
        return st.text_input("Stock ticker:", help=helpValue)

    def selectModel(self):
        return st.selectbox("Select prediction model:", MainApp.models)

    def createSideBar(self):
        with st.sidebar:
            st.write("CREATE STOCK PREDICTION")
            self.stock = self.selectStockTicker()
            self.model = self.selectModel()

    def drawMainStock(self):
        stock = DataFetcher(self.stock)
        return stock.getData("max")

if __name__ == '__main__':
    app = MainApp()
    app.createSideBar()
    st.dataframe(app.drawMainStock())