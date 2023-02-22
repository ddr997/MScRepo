import pandas as pd

from UI.central.Plot import Plot
class StateData:

    def __init__(self):

        # data fields
        self.stock = ""
        self.period = 0
        self.stockDataColumns = {}

        # indicators
        self.selectedIndicators = []

        # AI
        self.choosenModel = ""

        # object fields
        self.ticker = None
        self.dataFrame = pd.DataFrame()
        self.cleanDataFrame = pd.DataFrame()
        self.predictionDataFrame = pd.DataFrame()
        self.plot = Plot()

        #metrics
        self.metrics = dict()