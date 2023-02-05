import pandas as pd
from indicators.IndicatorsAbstract import IndicatorsAbstract
from indicators.trend.MA import MA

pd.set_option('display.max_colwidth', None)

class Ulcer(IndicatorsAbstract):

    def calculate(df: pd.DataFrame, period: int = 14):
        maxClose = df['Close'].rolling(window=period, min_periods=period).max()
        R = 100*(df['Close'] - maxClose)/maxClose # Percent-Drawdown
        ulcer = (R.pow(2, fill_value=0).rolling(window=period).sum()/period).pow(1/2, fill_value=0)
        df['Ulcer'] = ulcer
        return ulcer

if __name__ == '__main__':
    data = Ulcer.getDataFromTicker("ALE.WA", "1y")
    print(Ulcer.calculate(data, 14))
    print(data)
