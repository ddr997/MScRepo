import pandas as pd
from indicators.IndicatorsAbstract import IndicatorsAbstract

pd.set_option('display.max_colwidth', None)

class RSI(IndicatorsAbstract):
    def __init__(self):
        None

    def calculate(df: pd.DataFrame, periods: int, ema=True) -> pd.DataFrame:
        close_delta = df['Close'].diff() #diff from each day

        up = close_delta.clip(lower=0) #gains
        down = -1 * close_delta.clip(upper=0) #losses

        if ema == True:
            # Use exponential moving average
            ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
            ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        else:
            # Use simple moving average
            ma_up = up.rolling(window=periods, adjust=False).mean() #rolling equals to moving windows
            ma_down = down.rolling(window=periods, adjust=False).mean()

        rsi = ma_up / ma_down
        rsi = 100 - (100 / (1 + rsi))

        df['RSI'] = rsi
        return rsi

if __name__ == '__main__':
    data = RSI.getDataFromTicker("ALE.WA", "1y")
    print(data)
    print(RSI.calculate(data, 14))
    print(data)