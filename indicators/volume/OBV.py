import pandas as pd
import numpy as np
from indicators.IndicatorsAbstract import IndicatorsAbstract

pd.set_option('display.max_colwidth', None)

class OBV(IndicatorsAbstract):
    def __init__(self):
        None

    def calculate(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
        obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
        df['OBV'] = obv
        return obv

if __name__ == '__main__':
    data = OBV.getDataFromTicker("ALE.WA", "1y")
    print(data)
    print(OBV.calculate(data, 1))
    print(data)