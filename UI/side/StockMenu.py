import pandas
import streamlit as st
from Ticker import Ticker
from StateData import StateData

class StockMenu():

    def __init__(self, stateData: StateData):
        self.fetchSubmitted = False
        with st.sidebar:
            st.write("Fetch stock data")
            with st.form("stockForm"):
                self.selectStock(stateData)
                self.selectPeriod(stateData)
                self.selectStockDataFilter(stateData)
                self.fetchSubmitted = st.form_submit_button(label="Fetch Data")

        if(self.fetchSubmitted):
            self.createDataframeAndPlot(stateData)

    def selectStock(self, stateData):
        helpValue = "Model will be generated based on data fetched from Yahoo! Finance, " \
                    "based on a ticker name provided in a input."
        stateData.stock = st.text_input("Stock ticker:", value="ALE.WA", help=helpValue)
        return stateData.stock

    def selectPeriod(self, stateData):
        stateData.period = st.slider("Days to fetch from data source:", min_value=1, max_value=365 * 3, value=500)
        return stateData.period

    def selectStockDataFilter(self, stateData):
        labels = ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits")
        with st.expander("Select filter"):
            stateData.stockDataColumns = {label: st.checkbox(label, value=True) for label in labels[0:5]}
            stateData.stockDataColumns = [k for k in stateData.stockDataColumns.keys() if stateData.stockDataColumns.get(k)]
        return stateData.stockDataColumns

    def createDataframeAndPlot(self, stateData):
        stateData.ticker = Ticker(stateData.stock)
        stateData.dataFrame = stateData.ticker.getData(stateData.period)[stateData.stockDataColumns]
        stateData.cleanDataFrame = stateData.dataFrame.copy(deep=True)
        stateData.predictionDataFrame = pandas.DataFrame(stateData.dataFrame["Close"])
        stateData.plot.generateFigure(stateData.dataFrame)
        stateData.metrics = {}

        # save state
        st.session_state["state"] = stateData
        # rerun
        st.experimental_rerun()