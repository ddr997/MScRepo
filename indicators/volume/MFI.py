import pandas as pd
import numpy as np
from indicators.IndicatorsAbstract import IndicatorsAbstract

pd.set_option('display.max_colwidth', None)

class MFI(IndicatorsAbstract):
    def __init__(self):
        None

    def calculate(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        typical_price = (df['Close'] + df['High'] + df['Low']) / 3
        money_flow = typical_price * df['Volume']
        wasBigger = np.sign(typical_price.diff())

        positive_flow = []
        negative_flow = []

        for i in range(1, len(typical_price)):
            if wasBigger[i] == 1:
                positive_flow.append(money_flow[i - 1])
                negative_flow.append(0)

            elif wasBigger[i] == -1:
                negative_flow.append(money_flow[i - 1])
                positive_flow.append(0)

            else:
                positive_flow.append(0)
                negative_flow.append(0)

        pmf = pd.DataFrame(positive_flow, index=df.index[1:]).rolling(window=period, min_periods=period).sum()
        nmf = pd.DataFrame(negative_flow, index=df.index[1:]).rolling(window=period).sum()
        ratio = pmf/nmf

        mfi = pd.DataFrame(index=df.index)
        mfi = 100 - (100 / (1 + ratio))
        
        df['MFI'] = mfi
        return df['MFI']


if __name__ == '__main__':
    data = MFI.getDataFromTicker("ALE.WA", "1y")
    print(data)
    print(MFI.calculate(data, 14))
    print(data)