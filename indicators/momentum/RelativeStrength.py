import pandas as pd
from indicators.IndicatorsAbstract import IndicatorsAbstract
pd.set_option('display.max_colwidth', None)

class RelativeStrength(IndicatorsAbstract):
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        days = len(df.index)
        wig20 = RelativeStrength.getDataFromTicker("EURPLN=X", str(days)+"d")
        relativeStrength = df['Close']/wig20['Close']
        df["RelStr"] = relativeStrength
        return relativeStrength

if __name__ == '__main__':
    data = RelativeStrength.getDataFromTicker("ALE.WA", "1y")
    print(RelativeStrength.calculate(data))
    print(data)