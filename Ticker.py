import yfinance as yf
from pandas import DataFrame


class Ticker:
    def __init__(self, tickerIndex):
        self.ticker = yf.Ticker(tickerIndex)

    def getData(self, period: int) -> DataFrame:
        return self.ticker.history(
            period=str(period)+"d"
        )

