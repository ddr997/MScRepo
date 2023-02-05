from indicators.breadth.AdvanceDecline import *
from indicators.breadth.UDVR import *

from indicators.momentum.RelativeStrength import RelativeStrength
from indicators.momentum.ROC import ROC
from indicators.momentum.RSI import RSI
from indicators.momentum.Stochastic import Stochastic

from indicators.trend.ADX import ADX
from indicators.trend.MA import MA
from indicators.trend.MACD import MACD

from indicators.volatility.ATR import ATR
from indicators.volatility.CI import CI
from indicators.volatility.Ulcer import Ulcer

from indicators.volume.AD import AD
from indicators.volume.Chaikin import Chaikin
from indicators.volume.MFI import MFI
from indicators.volume.OBV import OBV

facade = {
    "AdvanceDecline": 0,
    "UDVR": 0,

    "RelativeStrength": RelativeStrength.calculate,
    "ROC": ROC.calculate,
    "RSI": RSI.calculate,
    "Stochastic": Stochastic.calculate,

    "ADX": ADX.calculate,
    "MA": MA.calculate,
    "MACD": MACD.calculate,

    "ATR": ATR.calculate,
    "CI": CI.calculate,
    "Ulcer": Ulcer.calculate,

    "AD": AD.calculate,
    "Chaikin": Chaikin.calculate,
    "MFI": MFI.calculate,
    "OBV": OBV.calculate
}

class IndicatorsFacade:
    indicatorsList = [i for i in facade.keys()]

    def map(key: str):
        return facade.get(key)