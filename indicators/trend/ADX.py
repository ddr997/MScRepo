import pandas as pd
from indicators.IndicatorsAbstract import IndicatorsAbstract
from indicators.volatility.ATR import ATR

pd.set_option('display.max_colwidth', None)

class ADX(IndicatorsAbstract):

    def calculate(df: pd.DataFrame, period: int):
        plus_dm = df["High"].diff()
        minus_dm = df["Low"].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        atr = ATR.calculate(df, period)

        plus_di = 100 * (plus_dm.ewm(alpha=1 / period).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha=1 / period).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (period - 1)) + dx) / period
        adx_smooth = adx.ewm(alpha=1 / period).mean()

        df["ADX"] = adx_smooth
        return plus_di, minus_di, adx_smooth

if __name__ == '__main__':
    data = ADX.getDataFromTicker("ALE.WA", "1y")
    print(ADX.calculate(data, 14))
    print(data)
