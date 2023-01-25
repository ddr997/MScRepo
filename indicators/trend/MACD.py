import pandas as pd
from indicators.IndicatorsAbstract import IndicatorsAbstract
from indicators.trend.MA import MA

pd.set_option('display.max_colwidth', None)

class MACD(IndicatorsAbstract):

    def calculate(dataFrame: pd.DataFrame, column = "Close") -> pd.DataFrame:
        df = dataFrame.copy()
        ema26 = MA.calculate(df, 26, column, EMA=True)
        ema12 = MA.calculate(df, 12, column, EMA=True)
        macd = ema12 - ema26

        macd_dF = macd.to_frame()
        signalLine = MA.calculate(macd_dF, 9, EMA=True)

        cd = macd - signalLine
        dataFrame['MACD'] = cd
        return cd


if __name__ == '__main__':
    data = MACD.getDataFromTicker("ALE.WA", "1y")
    print(data)
    print(MACD.calculate(data))
    print(data)