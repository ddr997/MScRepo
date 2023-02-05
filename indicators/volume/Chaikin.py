import pandas as pd
from indicators.IndicatorsAbstract import IndicatorsAbstract
from indicators.trend.MA import MA
from indicators.volume.AD import AD

pd.set_option('display.max_colwidth', None)

class Chaikin(IndicatorsAbstract):
    def __init__(self):
        None

    def calculate(dataFrame: pd.DataFrame) -> pd.DataFrame:
        df = dataFrame.copy()
        AD.calculate(df)
        macd3 = MA.calculate(df, period=3, EMA=True, column="AD")
        macd10 = MA.calculate(df, period=10, EMA=True, column="AD")
        chaikin = macd3 - macd10
        dataFrame['Chaikin'] = chaikin
        return chaikin

if __name__ == '__main__':
    data = Chaikin.getDataFromTicker("ALE.WA", "1y")
    print(data)
    print(Chaikin.calculate(data))
    print(data)