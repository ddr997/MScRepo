import pandas as pd
from indicators.IndicatorsAbstract import IndicatorsAbstract
from indicators.trend.MA import MA

pd.set_option('display.max_colwidth', None)

class ATR(IndicatorsAbstract):

    def calculate(df: pd.DataFrame, period: int = 14, returnTR = False):
        tr1 = pd.DataFrame(df['High'] - df['Low'])
        tr2 = pd.DataFrame(abs(df['High'] - df['Close'].shift(1)))
        tr3 = pd.DataFrame(abs(df['Low'] - df['Close'].shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        tr_dF = tr.to_frame()
        tr_dF.rename(columns={0: "TR"}, inplace=True)

        if returnTR: return tr_dF
        atr = MA.calculate(tr_dF, period, column="TR")

        df['ATR'] = atr
        return atr

if __name__ == '__main__':
    data = ATR.getDataFromTicker("ALE.WA", "1y")
    print(ATR.calculate(data, 14))
    print(data)
