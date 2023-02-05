import pandas as pd
from indicators.IndicatorsAbstract import IndicatorsAbstract

pd.set_option('display.max_colwidth', None)

class MA(IndicatorsAbstract):

    def calculate(df: pd.DataFrame, period: int = 14, column = "Close", EMA: bool = False) -> pd.DataFrame:
        if EMA:
            ema = df[column].ewm(span=period, adjust=True, min_periods=period).mean()
            df["EMA"] = ema
            return ema
        else:
            ma = df[column].rolling(window=period).mean() #rolling equals to moving windows
            df["SMA"] = ma
            return ma


if __name__ == '__main__':
    data = MA.getDataFromTicker("ALE.WA", "1y")
    print(data)
    print(MA.calculate(data, 14, EMA=True))
    print(data)