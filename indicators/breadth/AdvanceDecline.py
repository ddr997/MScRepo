import pandas as pd
from indicators.IndicatorsAbstract import IndicatorsAbstract
pd.set_option('display.max_colwidth', None)


WIG20 = ["ALE.WA", "ACP.WA", "CCC.WA", "CDR.WA", "CPS.WA", "DNP.WA", "JSW.WA", "KTY.WA", "KGH.WA", "KRU.WA", "LPP.WA", "MBK.WA", "OPL.WA", "PEO.WA", "PCO.WA", "PGE.WA", "PKN.WA", "PKO.WA", "PZU.WA", "SPL.WA"]

class AdvanceDecline(IndicatorsAbstract):
    def calculate(indexName: str = "WIG20") -> pd.DataFrame:
        if indexName == "WIG20":
            for i in WIG20:
                AdvanceDecline.getDataFromTicker()
        return

if __name__ == '__main__':
    data = AdvanceDecline.getDataFromTicker("ALE.WA", "1y")
    data2 = AdvanceDecline.getDataFromTicker("AMZN", "1y")
    print(AdvanceDecline.calculate(data, data2))
    print(data)