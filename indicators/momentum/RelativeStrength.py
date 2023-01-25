import pandas as pd
from indicators.IndicatorsAbstract import IndicatorsAbstract
pd.set_option('display.max_colwidth', None)

class RelativeStrength(IndicatorsAbstract):
    def __init__(self):
        None

    def calculate(df: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        relativeStrength = df['Close']/df2['Close']
        df["RelStr"] = relativeStrength
        return relativeStrength

if __name__ == '__main__':
    data = RelativeStrength.getDataFromTicker("ALE.WA", "1y")
    data2 = RelativeStrength.getDataFromTicker("AMZN", "1y")
    print(RelativeStrength.calculate(data, data2))
    print(data)