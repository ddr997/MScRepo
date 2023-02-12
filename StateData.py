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
        self.predictionFigure = None

        # object fields
        self.ticker = None
        self.dataFrame = None
        self.cleanDataFrame = None
        self.plot = Plot()