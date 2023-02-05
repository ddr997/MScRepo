import pandas as pd
import numpy as np
from indicators.IndicatorsAbstract import IndicatorsAbstract
from indicators.volatility.ATR import ATR

pd.set_option('display.max_colwidth', None)

class CI(IndicatorsAbstract):

    def calculate(df: pd.DataFrame, period: int = 14):
        tr = ATR.calculate(df, period, returnTR = True)
        sum_tr = tr.rolling(window=period, min_periods=period).sum()
        trueLow_n = df['Low'].rolling(window=period, min_periods=period).min()
        trueHigh_n = df['High'].rolling(window=period, min_periods=period).max()
        trueDiff = trueHigh_n - trueLow_n
        ci = 100 * np.log10((sum_tr['TR']/trueDiff)) / np.log10(period)
        df['CI'] = ci
        return ci

if __name__ == '__main__':
    data = CI.getDataFromTicker("ALE.WA", "1y")
    print(CI.calculate(data, 14))
    print(data)