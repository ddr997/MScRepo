from abc import abstractmethod

import pandas as pd
import yfinance as yf

class IndicatorsAbstract:

    @abstractmethod
    def calculate(self):
        pass

    @staticmethod
    def getDataFromTicker(stockIndex: str, period: str) -> pd.DataFrame:
        dataSource = yf.Ticker(stockIndex)
        data = dataSource.history(period=period)
        data = data.drop(columns=["Dividends", "Stock Splits"])
        return data

if __name__ == '__main__':
    data = yf.Ticker("EURPLN=X")
    print(data.history(period="1y"))