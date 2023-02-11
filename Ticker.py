import yfinance as yf
from pandas import DataFrame


class Ticker(yf.Ticker):
    def __init__(self, tickerIndex):
        self.ticker = yf.Ticker(tickerIndex)

    def getData(self, period: int) -> DataFrame:
        data = self.ticker.history(period=str(period)+"d")
        data.index.asfreq = 'B'
        return data

