import pandas as pd
from indicators.IndicatorsAbstract import IndicatorsAbstract
from indicators.trend.MA import MA

pd.set_option('display.max_colwidth', None)

class Ulcer(IndicatorsAbstract):

    def calculate(df: pd.DataFrame, period: int):
        return

if __name__ == '__main__':
    data = Ulcer.getDataFromTicker("ALE.WA", "1y")
    print(Ulcer.calculate(data, 14))
    print(data)