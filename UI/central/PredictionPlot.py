import plotly.express as px
import pandas as pd


class PredictionPlot:
    def __init__(self, df: pd.DataFrame):
        self.fig =  px.line(df, markers=True, title="Actual close price and predicted by models")
        self.fig.update_layout(
            height=500,
            width=1100,
            template='plotly',
            hovermode="x unified",
            yaxis = dict(title="Close Price", autorange=True, fixedrange=False),
            xaxis=dict(title="Date", autorange=True, fixedrange=False),
            legend_title = ""
        )

    def showPlot(self):
        self.fig.show()
        return

    def getPlot(self):
        return self.fig