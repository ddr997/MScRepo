import pandas as pd
import yfinance as yf

class IndicatorsAbstract:

    @staticmethod
    def getDataFromTicker(stockIndex: str, period: str) -> pd.DataFrame:
        dataSource = yf.Ticker(stockIndex)
        data = dataSource.history(period=period)
        data = data.drop(columns=["Dividends", "Stock Splits"])
        return data
