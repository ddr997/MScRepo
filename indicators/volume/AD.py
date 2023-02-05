import pandas as pd
from indicators.IndicatorsAbstract import IndicatorsAbstract

pd.set_option('display.max_colwidth', None)

class AD(IndicatorsAbstract):

    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close']))/(df['High'] - df['Low'])
        ad = (clv * df["Volume"]).cumsum()
        df['AD'] = ad
        return ad

if __name__ == '__main__':
    data = AD.getDataFromTicker("ALE.WA", "1y")
    print(data)
    print(AD.calculate(data))
    print(data)