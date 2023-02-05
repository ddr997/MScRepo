import streamlit as st
import plotly.express as px
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
        stateData.period = st.slider("Days to fetch from data source:", min_value=1, max_value=365 * 3, value=365)
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
        stateData.plot = px.line(
            stateData.dataFrame["Close"],
            x=stateData.dataFrame.index,
            y="Close",
            markers=True,
            title=f"{stateData.stock} closing price line chart"
        )
        stateData.plot.update_traces(line_color='#05F097', line_width=2, marker_size=1)

        # save state
        st.session_state["state"] = stateData
        # rerun
        st.experimental_rerun()