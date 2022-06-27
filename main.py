import pandas as pd
import plotly.express as px
import streamlit as st

from ModelCreator import ModelCreator
from Ticker import Ticker

pd.set_option('display.max_colwidth', None)


class MainApp:
    models = ["LSTM", "Linear Regression"]

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
            if "plotPred" in st.session_state.keys():
                st.plotly_chart(st.session_state.plotPred)
                col1, col2 = st.columns(2)
                col1.metric("RMSE", st.session_state.rmse)
                col2.metric("MAPE", st.session_state.mape)

    # Fetching methods
    def selectStock(self):
        helpValue = "Model will be generated based on data fetched from Yahoo! Finance, " \
                    "based on a ticker name provided in a input."
        self.stock = st.text_input("Stock ticker:", value="ALE.WA", help=helpValue)
        return self.stock

    def selectPeriod(self):
        self.periodToFetch = st.slider("Days to fetch from data source:",
                                       min_value=1, max_value=365 * 3, value=366)
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
            layers = ["LSTM", "Dense", "Dropout"]
            networkColumns = st.columns([1, 1])
            epochs = networkColumns[0].number_input("Epochs:", step=1, min_value=1, value=10)
            batchSize = networkColumns[1].number_input("Batch size:", step=1, min_value=1, value=32)
            windowSize = st.slider("Window size:", min_value=1, max_value=120, step=1, value=20)
            sender = ""

            with st.form("Add layer"):
                layerColumns = st.columns([1, 1])
                layer = layerColumns[0].selectbox("Layer:", layers)
                neurons = layerColumns[1].number_input("Neurons:", step=1, min_value=1)
                # returnSequence = st.selectbox("Return seq.", [True, False])
                if st.form_submit_button("Add layer"):
                    sender = f"\n{layer} {neurons}"

            if "concat" not in st.session_state:
                st.session_state.concat = "LSTM 96\nDropout 0.2\nLSTM 96\nDropout 0.2\nLSTM 96\nDropout 0.2\nDense 1"
            text = st.text_area("Layers parser", value=st.session_state.concat + sender)
            st.session_state.concat = text

            if st.button("Create prediction"):
                mc = ModelCreator(self.dataFetched)
                mc.parseLayers(text)
                st.session_state.plotPred, st.session_state.rmse, st.session_state.mape \
                    = mc.createLSTMPrediction(epochs, batchSize, windowSize)
                return 2

        if self.model == "Linear Regression":
            if st.button("Create prediction"):
                mc = ModelCreator(self.dataFetched)
                mc.createLinearRegressionPrediction()
            return 1

    # sideBar creator
    def createSideBar(self):
        fetchSubmitted = False
        modelSubmitted = False

        with st.sidebar:
            st.write("Fetch stock data")
            with st.form("stockForm"):
                self.selectStock()
                self.selectPeriod()
                self.selectFetchFilter()
                fetchSubmitted = st.form_submit_button(label="Fetch Data")

            st.write("Create prediction")
            self.selectModel()
            plotPred = self.selectionModelExpander()

        if fetchSubmitted:
            self.drawMainStockData()
        if plotPred == 2:
            st.experimental_rerun()
        if plotPred == 1:
            print(plotPred)

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
        self.graph.update_traces(line_color='#05F097', line_width=2, marker_size=1)

        # save state
        st.session_state["fetchedData"] = self
        # rerun
        st.experimental_rerun()
        return 1


if __name__ == '__main__':
    app = MainApp()
    app.createSideBar()
