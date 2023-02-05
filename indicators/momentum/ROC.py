import pandas as pd
from indicators.IndicatorsAbstract import IndicatorsAbstract

pd.set_option('display.max_colwidth', None)

class ROC(IndicatorsAbstract):
    def __init__(self):
        None

    def calculate(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
        N = df["Close"].diff(n)
        D = df["Close"].shift(n)
        roc = pd.Series(100* N/D, name="ROC")
        df["ROC"] = roc
        return roc

if __name__ == '__main__':
    data = ROC.getDataFromTicker("ALE.WA", "1y")
    print(data)
    print(ROC.calculate(data, 1))
    print(data)