from plotly.subplots import make_subplots
import plotly.graph_objects as go

class Plot:
    def __init__(self):
        self.fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01,
                                 specs=[[{}], [{}]])
        self.fig.update_xaxes(rangeslider_visible=False)
        self.fig.update_layout(
            height=700,
            width=1000,
            title_text="Charts",
            template='plotly',
            xaxis=dict(type="date"),
            yaxis=dict(autorange=True, fixedrange=False, domain=[0, 0.3]),
            xaxis2=dict(anchor="x"),
            yaxis2=dict(domain=[0.32, 1], title="Close Price", autorange=True, fixedrange=False,),
            hovermode="x unified"
        )
        self.fig.update_traces(
            marker=dict(color="LightSeaGreen"),
            line=dict(color="#ffe476")
        )
        self.fig.update_yaxes(fixedrange=False, autorange=True)
        self.fig.data = [] # clear data

        #indicator plot management
        self.firstTime = True
        self.previousIndicatorIndex = ""

    def generateFigure(self, dataFrame):
        self.fig.data = [] # clear data
        stockScatter = self.addStockTrace(dataFrame)
        stockCandles = self.addCandles(dataFrame)
        # predictionScatter = self.addPredictionTrace(dataFrame)
        return

    def addStockTrace(self, dataFrame):
        scatter = go.Scatter(
            x=dataFrame.index,
            y=dataFrame["Close"],
            name="Close",
            mode="lines",
            # marker=dict(color="LightSeaGreen", size=3),
            line=dict(color="RoyalBlue"),
        )
        self.fig.add_trace(scatter, row=2, col=1)
        return scatter

    def addIndicatorTrace(self, dataFrame, y="Volume"):
        if not self.firstTime:
            self.fig.data = [i for i in self.fig.data if i.name != self.previousIndicatorIndex]
        else:
            self.firstTime = False
        self.previousIndicatorIndex = y
        scatter = go.Scatter(
            x=dataFrame.index,
            y=dataFrame[y],
            name=y,
            mode="lines",
            # marker=dict(color="YellowGreen", size=3),
            line=dict(color="Orange"),
            xaxis="x2",
            yaxis="y2",
        )
        self.fig.add_trace(scatter, row=1, col=1)
        self.fig.update_layout(yaxis=dict(title=y))
        return

    def addPredictionTrace(self, dataFrame, y="PredictionTest"):
        scatter = go.Scatter(
                x=dataFrame.index,
                y=dataFrame["Open"],
                name=y,
                mode="lines",
                line=dict(color="red"),
                xaxis="x2",
                yaxis="y2",
            )
        self.fig.add_trace(scatter, row=2, col=1)
        return scatter

    def addCandles(self, df):
        candles = go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            yaxis="y",
            name="Candles",
            opacity=0.5
        )
        self.fig.add_trace(candles, row=2, col=1)
        return candles


if __name__ == "__main__":
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)
    #
    fig.add_trace(go.Scatter(x=[0, 1, 2], y=[10, 11, 12]),
                  row=3, col=1)
    #
    fig.add_trace(go.Scatter(x=[2, 3, 4], y=[100, 110, 120]),
                  row=2, col=1)
    #
    # fig.add_trace(go.Scatter(x=[3, 4, 5], y=[1000, 1100, 1200]),
    #               row=1, col=1)

    fig.update_layout(height=600, width=600,
                      title_text="Stacked Subplots with Shared X-Axes")
    fig.show()
