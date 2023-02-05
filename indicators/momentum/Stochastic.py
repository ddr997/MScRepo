import pandas as pd
import yfinance as yf
from indicators.IndicatorsAbstract import IndicatorsAbstract

pd.set_option('display.max_colwidth', None)

class Stochastic(IndicatorsAbstract):
    def __init__(self):
        None

    def calculate(dataFrame: pd.DataFrame, period: int = 14, D: int = 3) -> pd.DataFrame:
        df = dataFrame.copy(deep=True)
        # max/min of high/low in the period
        df['n_high'] = df['High'].rolling(period).max()
        df['n_low'] = df['Low'].rolling(period).min()
        df['%K'] = (df['Close'] - df['n_low']) * 100 / (df['n_high'] - df['n_low'])
        df['%D'] = df['%K'].rolling(D).mean()

        dataFrame["Stochastic.py"] = df["%D"]
        return df

if __name__ == '__main__':
    data = Stochastic.getDataFromTicker("ALE.WA", "1y")
    print(data)
    print(Stochastic.calculate(data, 14, 3))
    print(data)